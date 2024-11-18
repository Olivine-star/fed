import torch
import os
import numpy as np
import h5py
from utils.model_utils import get_dataset_name, RUNCONFIGS
import copy
import torch.nn.functional as F
import time
import torch.nn as nn
from utils.model_utils import get_log_path, METRICS


class Server:
    # 初始化服务器的主要属性，如数据集信息、训练参数、模型和用户信息等
    # 将传入的模型进行深拷贝，设置个性化标志，创建结果保存路径
    def __init__(self, args, model, seed):
        # Set up the main attributes
        self.dataset = args.dataset
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.K = args.K
        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        self.users = []
        self.selected_users = []
        self.num_users = args.num_users
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.personalized = 'pFed' in self.algorithm
        self.mode='partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        self.metrics = {key:[] for key in METRICS}
        self.timestamp = None
        self.save_path = args.result_path
        os.system("mkdir -p {}".format(self.save_path))

    # 初始化与集成学习相关的参数
    # 根据数据集名称，从全局配置中提取超参数，如学习率、批量大小等，打印相关参数
    def init_ensemble_configs(self):
        #### used for ensemble learning ####
        dataset_name = get_dataset_name(self.dataset)
        self.ensemble_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)#从 RUNCONFIGS 配置字典中读取当前数据集对应的键 ensemble_lr（集成学习的学习率）对应的值，如果找不到这个配置项，则使用默认的学习率 1e-4
        self.ensemble_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
        self.ensemble_epochs = RUNCONFIGS[dataset_name]['ensemble_epochs']
        self.num_pretrain_iters = RUNCONFIGS[dataset_name]['num_pretrain_iters']
        self.temperature = RUNCONFIGS[dataset_name].get('temperature', 1)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.ensemble_alpha = RUNCONFIGS[dataset_name].get('ensemble_alpha', 1)
        self.ensemble_beta = RUNCONFIGS[dataset_name].get('ensemble_beta', 0)
        self.ensemble_eta = RUNCONFIGS[dataset_name].get('ensemble_eta', 1)
        self.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.ensemble_train_loss = []
        self.n_teacher_iters = 5
        self.n_student_iters = 1
        print("ensemble_lr: {}".format(self.ensemble_lr) )
        print("ensemble_batch_size: {}".format(self.ensemble_batch_size) )
        print("unique_labels: {}".format(self.unique_labels) )


    # 判断当前算法是否是个性化联邦学习，判断当前算法是否为个性化联邦学习的方法是通过检查 self.algorithm 变量是否包含 'pFed' 或 'PerAvg' 这两个字符串。
    def if_personalized(self):
        return 'pFed' in self.algorithm or 'PerAvg' in self.algorithm


    # 判断当前算法是否为集成算法
    def if_ensemble(self):
        return 'FedE' in self.algorithm

    # 发送参数到所有或选定的用户，根据传输模式决定传输全部或部分参数
    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)#assert的作用是 进行条件检查，在程序运行过程中确保某些条件为真。如果条件为假，程序会立即抛出一个 AssertionError，并终止执行。
            users = self.selected_users
        #使用 for 循环 是为了 逐一遍历每个用户，并根据不同的模式为他们 发送相应的模型参数。
        for user in users:
            if mode == 'all': # share only subset of parameters
                user.set_parameters(self.model,beta=beta)
            else:  # share all parameters
                user.set_shared_parameters(self.model,mode=mode)

    # 聚合用户的模型参数到服务器，根据partial标志决定是否只聚合部分参数
    # 将用户参数按比例加到服务器模型中
    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio  # ration为当前 user的train sample/total train samples

    # 从选定用户聚合模型参数，重置服务器模型参数为零后加权聚合
    # 记录通信开销
    def aggregate_parameters(self, partial=False):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        if partial:  # 指示是否只对模型的某些参数进行聚合
            for param in self.model.get_shared_parameters():
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data)  # 将模型参数重置为全零
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples

        communication_overhead = 0
        for user in self.selected_users:
            communication_overhead += user.calculate_communication_data_size()
            self.add_parameters(user, user.train_samples / total_train,partial=partial)
        self.metrics['communication_overhead_upload'].append(communication_overhead)

    # 保存当前服务器模型到指定路径
    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    # 从磁盘加载服务器模型
    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    # 检查服务器模型文件是否存在
    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    # 随机选择指定数量的用户参与训练，支持返回用户索引
    # 当所有用户都被选中时，返回所有用户
    def select_users(self, round, num_users, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)

    # 初始化不同类型的损失函数
    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")#,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

    # 将训练过程中的指标数据保存为h5文件
    def save_results(self, args):
        alg = get_log_path(args, args.algorithm, self.seed, args.gen_batch_size)
        with h5py.File("./{}/{}.h5".format(self.save_path, alg), 'w') as hf:
            for key in self.metrics:
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()

    # 测试当前服务器模型在选定或所有用户上的表现
    # 返回每个用户的准确率和损失
    def test(self, selected=False):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    # 测试个性化模型在选定或所有用户上的表现
    def test_personalized_model(self, selected=True):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, ns, loss = c.test_personalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    # 评估个性化模型的准确率和损失，并保存结果
    def evaluate_personalized_model(self, selected=True, save=True):
        stats = self.test_personalized_model(selected=selected)
        test_ids, test_num_samples, test_tot_correct, test_losses = stats[:4]
        glob_acc = np.sum(test_tot_correct)*1.0/np.sum(test_num_samples)
        test_losses = [t.detach() for t in test_losses]
        test_loss = np.sum([x * y for (x, y) in zip(test_num_samples, test_losses)]).item() / np.sum(test_num_samples)
        if save:
            self.metrics['per_acc'].append(glob_acc)
            self.metrics['per_loss'].append(test_loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, test_loss))

    # 使用集成模型评估性能，聚合所有用户的logits并计算全局损失和准确率
    def evaluate_ensemble(self, selected=True):
        self.model.eval()
        users = self.selected_users if selected else self.users
        test_acc=0
        loss=0
        for x, y in self.testloaderfull:
            target_logit_output=0
            for user in users:
                # get user logit
                user.model.eval()
                user_result=user.model(x, logit=True)
                target_logit_output+=user_result['logit']
            target_logp=F.log_softmax(target_logit_output, dim=1)
            test_acc+= torch.sum( torch.argmax(target_logp, dim=1) == y ) #(torch.sum().item()
            loss+=self.loss(target_logp, y)
        loss = loss.detach().numpy()
        test_acc = test_acc.detach().numpy() / y.shape[0]
        self.metrics['glob_acc'].append(test_acc)
        self.metrics['glob_loss'].append(loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(test_acc, loss))

    # 评估全局模型性能，并记录评估结果
    def evaluate(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        test_losses = [t.detach() for t in test_losses]
        glob_loss = np.sum([x * y for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

