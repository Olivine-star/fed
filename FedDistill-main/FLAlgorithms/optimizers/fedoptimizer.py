from torch.optim import Optimizer

# 自定义的优化器类 pFedIBOptimizer，继承自 PyTorch 的 Optimizer 基类
class pFedIBOptimizer(Optimizer):
    # 初始化优化器，传入模型参数 params 和学习率 lr，默认为 0.01
    def __init__(self, params, lr=0.01):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:# 如果学习率 lr 小于 0，抛出异常
            raise ValueError("Invalid learning rate: {}".format(lr))
        # 将 lr 作为默认配置项，放入字典 defaults 中
        defaults=dict(lr=lr)
        # 调用父类 Optimizer 的初始化方法
        super(pFedIBOptimizer, self).__init__(params, defaults)

    # 优化器的 step 方法，用于更新模型的参数
    # apply 参数决定是否应用梯度更新；lr 用于设置自定义学习率；allow_unused 控制是否允许没有梯度的参数
    def step(self, apply=True, lr=None, allow_unused=False):
        grads = [] # 用于保存计算出的梯度
        # apply gradient to model.parameters, and return the gradients
        for group in self.param_groups:
            # 遍历每个参数 p
            for p in group['params']:
                # 如果该参数没有梯度，并且 allow_unused 设置为允许没有梯度，则跳过此参数
                if p.grad is None and allow_unused:
                    continue
                # 将参数的梯度保存到 grads 列表中
                grads.append(p.grad.data)
                # 如果 apply 为 True，则根据梯度和学习率更新参数
                if apply:
                    # 如果未提供自定义学习率，则使用默认的学习率更新参数
                    if lr == None:
                        p.data= p.data - group['lr'] * p.grad.data
                    # 否则，使用提供的自定义学习率来更新参数
                    else:
                        p.data=p.data - lr * p.grad.data
        return grads # 返回梯度列表

    # 自定义方法 apply_grads，用于手动应用传入的梯度来更新模型的参数
    # beta 参数允许通过不同的缩放因子来应用梯度
    def apply_grads(self, grads, beta=None, allow_unused=False):
        #apply gradient to model.parameters
        i = 0 # 用于跟踪 grads 列表中的索引
        # 遍历每个参数组
        for group in self.param_groups:
            # 遍历每个参数 p
            for p in group['params']:
                # 如果该参数没有梯度，并且 allow_unused 设置为允许没有梯度，则跳过此参数
                if p.grad is None and allow_unused:
                    continue
                # 根据是否提供 beta 来决定使用默认学习率还是 beta 值更新参数
                p.data= p.data - group['lr'] * grads[i] if beta == None else p.data - beta * grads[i]
                i += 1 # 更新 grads 列表中的索引
        return
