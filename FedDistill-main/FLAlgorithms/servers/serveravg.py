from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
# Implementation for FedAvg Server
import time

# 实现 FedAvg 服务器端算法的类, FedAvg 类继承自 Server 类
class FedAvg(Server):
    def __init__(self, args, model, seed):
        # 初始化父类 Server 的参数,super() 是用来引用父类的一个特殊函数，可以用来调用父类的方法。特别是在子类的 __init__ 方法中调用 super().__init__(...)，以确保父类的初始化代码能被执行。
        super().__init__(args, model, seed)

        # Initialize data for all  users
        data = read_data(args.dataset)# 读取指定数据集的数据
        total_users = len(data[0])# 获取总用户数,这里不用self.,因为total_users 只在 __init__ 方法中使用，并没有在类的其他方法中使用或共享，因此定义为局部变量即可，不需要使用 self
        self.use_adam = 'adam' in self.algorithm.lower()# 判断是否使用 Adam 优化器（使用 self 是为了将 use_adam 定义为当前类实例的属性。
        # 使用 self.use_adam 将数据绑定到类的实例，使得其他方法可以方便地访问 use_adam 的值。如果不用 self，use_adam 就只是一个局部变量，在当前方法执行结束后就会被销毁，无法在其他地方使用。）
        print("Users in total: {}".format(total_users))# 打印总用户数

        # 遍历所有用户数据并初始化用户对象
        for i in range(total_users):
            # 读取第 i 个用户的训练数据和测试数据
            id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
            # 创建 UserAVG 对象表示用户，并将其添加到服务器的用户列表中,UserAVG 是一个类，通常用于表示联邦学习中的用户。
            user = UserAVG(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user)#user 对象被添加到服务器的 users 列表中，表示该用户已经注册并准备参与联邦学习。
            # 累计训练样本数
            self.total_train_samples += user.train_samples

        # 打印当前选择的用户数量和总用户数量
        print("Number of users / total users:",args.num_users, " / " ,total_users)
        print("Finished creating FedAvg server.") # 创建 FedAvg 服务器完成

    # 联邦学习的训练函数
    def train(self, args):
        # 迭代进行联邦学习的全局轮数,self.num_glob_iters 表示全局训练的轮数。这个参数定义了要进行多少次全局训练
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            # 随机选择用户进行本轮训练
            self.selected_users = self.select_users(glob_iter,self.num_users)
            # 将全局模型参数发送给选中的用户
            self.send_parameters(mode=self.mode)
            # 在当前轮次开始训练前进行模型评估
            self.evaluate()
            self.timestamp = time.time() # log user-training start time
            # 对选定的用户进行本轮训练
            for user in self.selected_users: # allow selected users to train
                    user.train(glob_iter, personalized=self.personalized) #* user.train_samples
            # 记录用户训练结束时间
            curr_timestamp = time.time() # log  user-training end time
            # 计算并记录每个用户的平均训练时间
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)

            # 如果使用个性化模型，则评估个性化模型
            # Evaluate selected user
            if self.personalized:
                # Evaluate personal model on user for each iteration
                print("Evaluate personal model\n")
                self.evaluate_personalized_model()

            # 记录服务器聚合参数开始时间
            self.timestamp = time.time()  # log server-agg start time
            # 聚合用户模型参数
            self.aggregate_parameters()
            # 记录服务器聚合参数结束时间
            curr_timestamp=time.time()  # log  server-agg end time
            # 计算并记录聚合时间
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
        # 保存训练结果和模型
        self.save_results(args)
        self.save_model()