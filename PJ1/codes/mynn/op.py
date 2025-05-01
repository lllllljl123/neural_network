from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"[{type(self).__name__}] forward() not implemented.")

    @abstractmethod
    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"[{type(self).__name__}] forward() not implemented.")


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.input.T, grad) / self.input.shape[0]
        self.grads['b'] = np.mean(grad, axis=0, keepdims=True)

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        return np.dot(grad, self.W.T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class Dropout(Layer):
    def __init__(self, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.mask = None
        self.training = True
        self.optimizable = False

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.drop_prob, size=x.shape) / (1 - self.drop_prob)
            return x * self.mask
        else:
            return x

    def backward(self, grad):
        return grad * self.mask if self.training else grad

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.shape)

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        # Xavier 初始化
        scale = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
        self.W = np.random.uniform(-scale, scale, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels, 1))

        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        B, C, H, W = X.shape
        K = self.kernel_size
        out_h = (H - K + 2 * self.padding) // self.stride + 1
        out_w = (W - K + 2 * self.padding) // self.stride + 1
        out = np.zeros((B, self.out_channels, out_h, out_w))

        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X

        for b in range(B):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = X_padded[b, :, h_start:h_start + K, w_start:w_start + K]
                        out[b, oc, i, j] = np.sum(region * self.W[oc]) + self.b[oc]

        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        B, C_in, H, W = X.shape
        K = self.kernel_size
        C_out, _, _, _ = self.W.shape
        _, _, out_H, out_W = grads.shape

        # Initialize gradient buffers
        dX = np.zeros_like(X)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for b in range(B):
            for oc in range(C_out):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = X[b, :, h_start:h_start + K, w_start:w_start + K]  # shape: [C_in, K, K]

                        # grad w.r.t weights
                        dW[oc] += grads[b, oc, i, j] * region
                        # grad w.r.t bias
                        db[oc] += grads[b, oc, i, j]
                        # grad w.r.t input
                        dX[b, :, h_start:h_start + K, w_start:w_start + K] += grads[b, oc, i, j] * self.W[oc]

        self.grads['W'] = dW / B
        self.grads['b'] = db / B

        # Weight decay
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MaxPool2x2(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.mask = None
        self.optimizable = False

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input = x
        B, C, H, W = x.shape
        out_H, out_W = H // 2, W // 2
        out = np.zeros((B, C, out_H, out_W))
        self.mask = np.zeros_like(x)

        for b in range(B):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start, w_start = i * 2, j * 2
                        patch = x[b, c, h_start:h_start+2, w_start:w_start+2]
                        max_val = np.max(patch)
                        out[b, c, i, j] = max_val
                        # 保存 mask：最大值位置为 1，其余为 0
                        mask = (patch == max_val)
                        self.mask[b, c, h_start:h_start+2, w_start:w_start+2] = mask
        return out

    def backward(self, grad):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input)
        out_H, out_W = grad.shape[2], grad.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start, w_start = i * 2, j * 2
                        grad_input[b, c, h_start:h_start+2, w_start:w_start+2] += \
                            grad[b, c, i, j] * self.mask[b, c, h_start:h_start+2, w_start:w_start+2]
        return grad_input


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.optimizable = False
        self.has_softmax = True
    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.predicts = predicts
        self.labels = labels

        # Softmax
        exp_preds = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        self.probs = probs

        log_probs = -np.log(probs[np.arange(len(labels)), labels] + 1e-9)
        loss = np.mean(log_probs)
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        batch_size = self.labels.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.labels] -= 1
        grad /= batch_size
        self.grads = grad
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition