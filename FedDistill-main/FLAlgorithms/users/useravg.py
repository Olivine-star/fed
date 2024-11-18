
#这段代码负责联邦学习中用户端的模型训练和参数更新。
import torch
from FLAlgorithms.users.userbase import User

#初始化用户的FedAvg对象，继承自基础User类，设置训练数据、测试数据和模型。
class UserAVG(User):
    def __init__(self,  args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

    #更新标签计数器。根据输入的标签和对应计数，更新self.label_counts。
    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    #清空并重置标签计数器。删除当前的self.label_counts，并重新初始化。
    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    # 执行本地模型的训练，并在每一轮训练后克隆模型参数到本地模型。
    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        self.clean_up_counts()
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for i in range(self.K):
                result =self.get_next_train_batch(count_labels=count_labels)
                X, y = result['X'], result['y']
                if count_labels:
                    self.update_label_counts(result['labels'], result['counts'])

                self.optimizer.zero_grad()
                output=self.model(X)['output']
                loss=self.loss(output, y)
                loss.backward()
                self.optimizer.step()#self.plot_Celeb)

            # local-model <=== self.model
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
            # local-model ===> self.model
            #self.clone_model_paramenter(self.local_model, self.model.parameters())
        if lr_decay:
            self.lr_scheduler.step(glob_iter)
