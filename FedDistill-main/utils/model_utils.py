'''

这段代码的主要作用是为联邦学习和生成模型的训练过程提供数据加载、模型创建、参数更新和损失计算的功能。具体包括：

数据处理：根据数据集的不同格式（如EMnist、Mnist、CelebA），读取、转换并加载训练、测试和代理数据。
模型创建与参数更新：根据指定的数据集和算法，初始化模型，并提供多种方式更新模型参数，如 Polyak 和 Meta 更新。
损失计算：实现了 Moreau 和 L2 等损失函数的计算，用于正则化和参数优化。
联邦学习支持：为联邦学习中的多个客户端提供数据聚合、训练数据加载等功能，支持个性化模型和生成器的训练。
'''

import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import random
import numpy as np
from FLAlgorithms.trainmodel.models import Net
from FLAlgorithms.trainmodel.models_v2 import SimpleNet

from torch.utils.data import DataLoader
from utils.model_config import *
METRICS = ['glob_acc', 'per_acc', 'glob_loss', 'per_loss', 'user_train_time', 'server_agg_time', 'communication_overhead_upload']


# 根据数据集名称生成训练、测试和代理数据的路径
# 根据数据集的不同（如EMnist、Mnist、CelebA），使用不同的文件路径结构，并返回相关路径
def get_data_dir(dataset):
    if 'EMnist' in dataset:
        #EMnist-alpha0.1-ratio0.1-0-letters
        dataset_=dataset.replace('alpha', '').replace('ratio', '').split('-')
        alpha, ratio =dataset_[1], dataset_[2]
        types = 'letters'
        path_prefix = os.path.join('data', 'EMnist', f'u20-{types}-alpha{alpha}-ratio{ratio}')
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')
        proxy_data_dir = 'data/proxy_data/emnist-n10/'

    elif 'Mnist' in dataset:
        dataset_=dataset.replace('alpha', '').replace('ratio', '').split('-')
        alpha, ratio=dataset_[1], dataset_[2]
        #path_prefix=os.path.join('data', 'Mnist', 'u20alpha{}min10ratio{}'.format(alpha, ratio))
        path_prefix=os.path.join('data', 'Mnist', 'u20c10-alpha{}-ratio{}'.format(alpha, ratio))
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')
        proxy_data_dir = 'data/proxy_data/mnist-n10/'

    elif 'celeb' in dataset.lower():
        dataset_ = dataset.lower().replace('user', '').replace('agg','').split('-')
        user, agg_user = dataset_[1], dataset_[2]
        path_prefix = os.path.join('data', 'CelebA', 'user{}-agg{}'.format(user,agg_user))
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')
        proxy_data_dir=os.path.join('/user500/', 'proxy')

    else:
        raise ValueError("Dataset not recognized.")
    return train_data_dir, test_data_dir, proxy_data_dir

# 读取并解析给定数据集的训练和测试数据
# 根据不同的目录结构（如EMnist、Mnist、CelebA），读取对应的 .json 或 .pt 文件，提取用户和用户数据，并返回客户端信息、分组信息、训练数据、测试数据以及代理数据
def read_data(dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_data_dir, test_data_dir, proxy_data_dir = get_data_dir(dataset)
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    proxy_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') or f.endswith(".pt")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))
        test_data.update(cdata['user_data'])


    if proxy_data_dir and os.path.exists(proxy_data_dir):
        proxy_files=os.listdir(proxy_data_dir)
        proxy_files=[f for f in proxy_files if f.endswith('.json') or f.endswith(".pt")]
        for f in proxy_files:
            file_path=os.path.join(proxy_data_dir, f)
            if file_path.endswith(".pt"):
                with open(file_path, 'rb') as inf:
                    cdata=torch.load(inf)
            elif file_path.endswith(".json"):
                with open(file_path, 'r') as inf:
                    cdata=json.load(inf)
            else:
                raise TypeError("Data format not recognized: {}".format(file_path))
            proxy_data.update(cdata['user_data'])

    return clients, groups, train_data, test_data, proxy_data

# 将代理数据转换为适合训练的数据格式，并创建数据加载器
# 使用 DataLoader 将代理数据进行批次化处理，以便后续训练使用
def read_proxy_data(proxy_data, dataset, batch_size):
    X, y=proxy_data['x'], proxy_data['y']
    X, y = convert_data(X, y, dataset=dataset)
    dataset = [(x, y) for x, y in zip(X, y)]
    proxyloader = DataLoader(dataset, batch_size, shuffle=True)
    iter_proxyloader = iter(proxyloader)
    return proxyloader, iter_proxyloader


# 聚合多个客户端的数据，转换为适合训练的格式，并创建数据加载器
# 将每个客户端的数据（特征和标签）组合为一个大数据集，并返回数据加载器和唯一标签
def aggregate_data_(clients, dataset, dataset_name, batch_size):
    combined = []
    unique_labels = []
    for i in range(len(dataset)):
        id = clients[i]
        user_data = dataset[id]
        X, y = convert_data(user_data['x'], user_data['y'], dataset=dataset_name)
        combined += [(x, y) for x, y in zip(X, y)]
        unique_y=torch.unique(y)
        unique_y = unique_y.detach().numpy()
        unique_labels += list(unique_y)

    data_loader=DataLoader(combined, batch_size, shuffle=True)
    iter_loader=iter(data_loader)
    return data_loader, iter_loader, unique_labels

# 聚合用户的测试数据，并返回数据加载器和唯一标签
# 主要用于测试时，将多个用户的数据合并后用于评估
def aggregate_user_test_data(data, dataset_name, batch_size):
    clients, loaded_data=data[0], data[3]
    data_loader, _, unique_labels=aggregate_data_(clients, loaded_data, dataset_name, batch_size)
    return data_loader, np.unique(unique_labels)

# 聚合用户的训练数据，并返回数据加载器、迭代器和唯一标签
# 主要用于训练时，将多个用户的数据合并后用于模型训练
def aggregate_user_data(data, dataset_name, batch_size):
    # data contains: clients, groups, train_data, test_data, proxy_data
    clients, loaded_data = data[0], data[2]
    data_loader, data_iter, unique_labels = aggregate_data_(clients, loaded_data, dataset_name, batch_size)
    return data_loader, data_iter, np.unique(unique_labels)

# 将原始数据（X 和 y）转换为 PyTorch 张量格式
# 根据不同的数据集名称，如 CelebA，处理特征和标签的格式
def convert_data(X, y, dataset=''):
    if not isinstance(X, torch.Tensor):
        if 'celeb' in dataset.lower():
            X=torch.Tensor(X).type(torch.float32).permute(0, 3, 1, 2)
            y=torch.Tensor(y).type(torch.int64)

        else:
            X=torch.Tensor(X).type(torch.float32)
            y=torch.Tensor(y).type(torch.int64)
    return X, y

# 根据用户索引读取单个用户的训练和测试数据，并进行格式转换
# 可选地统计标签信息并返回标签统计数据
def read_user_data(index, data, dataset='', count_labels=False):
    #data contains: clients, groups, train_data, test_data, proxy_data(optional)
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train = convert_data(train_data['x'], train_data['y'], dataset=dataset)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    X_test, y_test = convert_data(test_data['x'], test_data['y'], dataset=dataset)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts
        return id, train_data, test_data, label_info
    return id, train_data, test_data

# 根据数据集名称返回标准化的名称
# 支持 'celeb'、'emnist' 和 'mnist'，不支持的会抛出错误

def get_dataset_name(dataset):
    dataset=dataset.lower()
    passed_dataset=dataset.lower()
    if 'celeb' in dataset:
        passed_dataset='celeb'
    elif 'emnist' in dataset:
        passed_dataset='emnist'
    elif 'mnist' in dataset:
        passed_dataset='mnist'
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    return passed_dataset

# 根据数据集名称和算法创建模型实例
# 返回一个简单网络（SimpleNet）的模型对象
def create_model(model, dataset, algorithm):
    passed_dataset = get_dataset_name(dataset)
    # model = Net(passed_dataset, model), model
    model = SimpleNet(passed_dataset, model), model
    return model

# 执行 Polyak 移动，将目标参数移动到当前参数方向上
# 逐个参数地按比例更新目标参数，使其靠近当前参数
def polyak_move(params, target_params, ratio=0.1):
    for param, target_param in zip(params, target_params):
        param.data=param.data - ratio * (param.clone().detach().data - target_param.clone().detach().data)

# 执行 Meta 更新，将目标参数移动到当前参数方向上，并按比例更新
def meta_move(params, target_params, ratio):
    for param, target_param in zip(params, target_params):
        target_param.data = param.clone().data + ratio * (target_param.clone().data - param.clone().data)

# 计算 Moreau 损失，表示参数与正则化参数的平方差
# 对每个参数求平方差，并取平均值作为损失
def moreau_loss(params, reg_params):
    # return 1/T \sum_i^T |param_i - reg_param_i|^2
    losses = []
    for param, reg_param in zip(params, reg_params):
        losses.append( torch.mean(torch.square(param - reg_param.clone().detach())) )
    loss = torch.mean(torch.stack(losses))
    return loss

def l2_loss(params):
    losses = []
    for param in params:
        losses.append( torch.mean(torch.square(param)))
    loss = torch.mean(torch.stack(losses))
    return loss

# 更新快速权重参数，使用梯度和学习率进行更新
# 对每个梯度和对应的权重进行更新，限制梯度范围在 -10 到 10 之间
def update_fast_params(fast_weights, grads, lr, allow_unused=False):
    """
    Update fast_weights by applying grads.
    :param fast_weights: list of parameters.
    :param grads: list of gradients
    :param lr:
    :return: updated fast_weights .
    """
    for grad, fast_weight in zip(grads, fast_weights):
        if allow_unused and grad is None: continue
        grad=torch.clamp(grad, -10, 10)
        fast_weight.data = fast_weight.data.clone() - lr * grad
    return fast_weights

# 初始化具有指定关键词（如 'encode'）的模型参数，并将它们设为需要计算梯度
# 从模型的命名层中筛选包含指定关键词的参数，返回一个字典
def init_named_params(model, keywords=['encode']):
    named_params={}
    #named_params_list = []
    for name, params in model.named_layers.items():
        if any([key in name for key in keywords]):
            named_params[name]=[param.clone().detach().requires_grad_(True) for param in params]
            #named_params_list += named_params[name]
    return named_params#, named_params_list

# 根据输入参数生成日志文件路径
# 包括数据集名称、算法名称、学习率、用户数量、批量大小、本地轮次等信息，并返回路径

def get_log_path(args, algorithm, seed, gen_batch_size=32):
    alg=args.dataset + "_" + algorithm
    alg+="_" + str(args.learning_rate) + "_" + str(args.num_users)
    alg+="u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    alg=alg + "_" + str(seed)
    if 'FedGen' in algorithm: # to accompany experiments for author rebuttal
        alg += "_embed" + str(args.embedding)
        if int(gen_batch_size) != int(args.batch_size):
            alg += "_gb" + str(gen_batch_size)
    return alg