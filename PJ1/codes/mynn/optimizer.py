from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
        self.lr = init_lr
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                if layer.weight_decay:
                    layer.W *= (1 - self.init_lr * layer.weight_decay_lambda)
                layer.W -= self.lr * layer.grads['W']
                layer.b -= self.lr * layer.grads['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu = 0.9):
        super().__init__(init_lr, model)
        self.lr = init_lr
        self.beta = mu
        self.prev_weights = {}

        # 初始化 prev_weights（上一轮的 w）
        for layer in self.model.layers:
            if layer.optimizable:
                self.prev_weights[layer] = {
                    'W': np.copy(layer.W),
                    'b': np.copy(layer.b)
                }
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                # 保存当前权重
                w_prev = self.prev_weights[layer]['W']
                b_prev = self.prev_weights[layer]['b']

                w_now = layer.W
                b_now = layer.b

                # 计算动量更新
                delta_w = w_now - w_prev
                delta_b = b_now - b_prev

                layer.W = layer.W - self.init_lr * layer.grads['W'] + self.beta * delta_w
                layer.b = layer.b - self.init_lr * layer.grads['b'] + self.beta * delta_b

                # 同步 params
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b

                # 更新 prev_weights
                self.prev_weights[layer]['W'] = np.copy(w_now)
                self.prev_weights[layer]['b'] = np.copy(b_now)