import numpy as np
import cv2 as cv
import visualization as vis
import os
from datetime import datetime
from run import Run
from data import MNISTDataset, one_hot_encode, compute_accuracy, normalize
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

class Module():
    def forward(self, input:np.ndarray):
        raise NotImplementedError
    
    def backward(self, grad_output:np.ndarray):
        raise NotImplementedError

class ReLU(Module):
    def forward(self, input:np.ndarray):
        self.input = input.copy()
        output = np.maximum(0, input)
        return output
    
    def backward(self, grad_output:np.ndarray):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class DenseLayer(Module):
    def __init__(
            self,
            in_features:int,
            out_features:int
    ):
        self.in_features = in_features
        self.out_features = out_features
        # He initialization
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.biases = np.zeros((out_features))

    def forward(self, input:np.ndarray):
        self.input = input.copy()
        output = input @ self.weights + self.biases
        return output
    
    def backward(self, grad_output:np.ndarray):
        grad_input = grad_output @ self.weights.T
        grad_weights = self.input.T @ grad_output
        grad_biases = np.sum(grad_output, axis=0)

        self.weights -= grad_weights
        self.biases -= grad_biases

        return grad_input

class Flatten(Module):
    def forward(self, input:np.ndarray):
        self.input_shape = input.shape
        batch_size = input.shape[0]
        output = input.reshape(batch_size, -1)
        return output
    
    def backward(self, grad_output:np.ndarray):
        grad_input = grad_output.reshape(self.input_shape)
        return grad_input

class Conv(Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size:int
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # He initialization
        fan_in = in_channels * kernel_size * kernel_size
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)

    def forward(self, input:np.ndarray):
        self.input = input.copy()
        batch_size, _, height, width = input.shape
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1
        patches = sliding_window_view(input, (self.kernel_size, self.kernel_size), axis=(2, 3))
        output = np.tensordot(
            patches,
            self.kernels,
            axes=([1, 4, 5], [1, 2, 3])
        )
        output = np.transpose(output, (0, 3, 1, 2))

        return output
    
    def backward(self, grad_output:np.ndarray):
        _, _, height, width = self.input.shape
        patches = sliding_window_view(self.input, (self.kernel_size, self.kernel_size), axis=(2, 3))

        grad_kernels = np.tensordot(
            grad_output,
            patches,
            axes=([0, 2, 3], [0, 2, 3])
        )

        pad = self.kernel_size - 1
        grad_output_padded = np.pad(grad_output, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        grad_windows = sliding_window_view(grad_output_padded, (self.kernel_size, self.kernel_size), axis=(2, 3))
        kernels_flipped = np.flip(self.kernels, axis=(2, 3))
        grad_input = np.tensordot(
            grad_windows,
            kernels_flipped,
            axes=([1, 4, 5], [0, 2, 3])
        )
        grad_input = np.transpose(grad_input, (0, 3, 1, 2))

        self.kernels -= grad_kernels

        return grad_input

class MaxPool(Module):
    def __init__(self, kernel_size:int, stride:int):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input:np.ndarray):
        self.input = input.copy()
        batch_size, channels, height, width = input.shape
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, channels, output_height, output_width))

        for i in range(output_height):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size

            for j in range(output_width):
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                region = input[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(region, axis=(2, 3))

        return output
    
    def backward(self, grad_output:np.ndarray):
        grad_input = np.zeros_like(self.input)

        for i in range(grad_output.shape[2]):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size

            for j in range(grad_output.shape[3]):
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                region = self.input[:, :, h_start:h_end, w_start:w_end]
                max_value = np.max(region, axis=(2, 3), keepdims=True)
                mask = region == max_value
                grad = grad_output[:, :, i, j][:, :, None, None]
                grad_input[:, :, h_start:h_end, w_start:w_end] += mask * grad

        return grad_input

class NeuralNetwork(Module):
    def __init__(self, modules:list[Module]):
        self.modules = modules

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input:np.ndarray):
        for module in self.modules:
            input = module.forward(input)
        return input
    
    def backward(self, grad_output:np.ndarray):
        for module in reversed(self.modules):
            grad_output = module.backward(grad_output)
        return grad_output
    
    def save(self, fname:str):
        np.savez(fname, *[module.__dict__ for module in self.modules])

    def load(self, fname:str):
        if os.path.exists(fname):
            data = np.load(fname, allow_pickle=True)
            for module, module_data in zip(self.modules, data.values()):
                module.__dict__.update(module_data.item())

class CrossEntropyLoss():
    def __init__(self, reduction = 'sum'):
        self.reduction = reduction

    def forward(self, input:np.ndarray, target:np.ndarray):
        self.input = input.copy()
        self.target = target.copy()
        batch_size = input.shape[0]
        loss = -np.sum(target * np.log(input + 1e-15)) / batch_size

        if self.reduction == 'mean':
            n_outputs = input.shape[1]
            return loss / n_outputs

        return loss
    
    def backward(self):
        if self.reduction == 'mean':
            n_outputs = self.input.shape[1]
            grad_input = -self.target / (self.input + 1e-15) / n_outputs
            return grad_input
        else:
            grad_input = -self.target / (self.input + 1e-15)
            return grad_input
        
        # batch_size = self.input.shape[0]
        # grad_input = -self.target / (self.input + 1e-15) / batch_size
        # return grad_input

class Softmax(Module):
    def forward(self, input:np.ndarray):
        self.input = input.copy()
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad_output:np.ndarray):
        dot = np.sum(grad_output * self.output, axis=1, keepdims=True)
        grad_input = self.output * (grad_output - dot)
        return grad_input
    
class CrossEntropyLossWithSoftmax(Module):
    """
    This module combines the softmax activation and the cross-entropy loss into a single module for numerical stability and computational efficiency.
    """
    def forward(self, input:np.ndarray, target:np.ndarray):
        self.input = input.copy()
        self.target = target.copy()
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        batch_size = input.shape[0]
        loss = -np.sum(target * np.log(self.output + 1e-15)) / batch_size
        return loss
    
    def backward(self):
        batch_size = self.input.shape[0]
        grad_input = (self.output - self.target) / batch_size
        return grad_input

if __name__ == "__main__":
    RUN_ID              = 'ConvModel-v1'
    EPOCHS              = 10
    LEARNING_RATE       = 1e-3
    BATCH_SIZE          = 128
    NUM_CLASSES         = 10
    PLOT_STEPS          = 100
    BATCHES_PER_EPOCH   = 500000
    MAVG_WINDOWS        = 500

    ds_train = MNISTDataset(split = 'train', batch_size=BATCH_SIZE)
    ds_val = MNISTDataset(split = 'val', batch_size=BATCH_SIZE)

    run = Run(run_id=RUN_ID, mavg_window=len(ds_train))
    #vis.plot_images(ds_train.images, ds_train.labels, num_images=40)

    # model = NeuralNetwork(modules=[
    #     Conv(in_channels=1, out_channels=16, kernel_size=3),
    #     ReLU(),
    #     MaxPool(kernel_size=2, stride=2),
    #     Conv(in_channels=16, out_channels=32, kernel_size=3),
    #     ReLU(),
    #     MaxPool(kernel_size=2, stride=2),

    #     Conv(in_channels=32, out_channels=64, kernel_size=3),
    #     ReLU(),
    #     MaxPool(kernel_size=2, stride=2),

    #     Flatten(),
    #     DenseLayer(in_features=64*1*1, out_features=NUM_CLASSES),
    #     #Softmax()
    # ])
    model = NeuralNetwork(modules=[
        Conv(in_channels=1, out_channels=32, kernel_size=3),
        ReLU(),
        MaxPool(kernel_size=2, stride=2),

        Conv(in_channels=32, out_channels=64, kernel_size=3),
        ReLU(),
        MaxPool(kernel_size=2, stride=2),

        Flatten(),
        DenseLayer(in_features=64*5*5, out_features=NUM_CLASSES),
        #Softmax()
    ])
    model.load(f"{run.run_dir}/model.npz")

    criterion = CrossEntropyLossWithSoftmax()

    start = datetime.now()
    for epoch in range(EPOCHS):
        ds_train.reset()
        n_batches_train = BATCHES_PER_EPOCH if BATCHES_PER_EPOCH < len(ds_train) else len(ds_train)
        prog_train = tqdm(ds_train, total=n_batches_train, leave=False, desc='train')
        for i, (X, Y) in enumerate(prog_train):
            if i >= n_batches_train:
                break
            X = normalize(X)
            Y = one_hot_encode(Y, num_classes=NUM_CLASSES)
            Y_hat = model(X)

            loss = criterion.forward(Y_hat, Y)
            accuracy = compute_accuracy(Y_hat, Y)

            grad_loss = criterion.backward()
            model.backward(grad_loss * LEARNING_RATE)

            run.append('loss', 'train', loss, step_epoch=(epoch * n_batches_train + i) / n_batches_train)
            run.append('accuracy', 'train', accuracy, step_epoch=(epoch * n_batches_train + i) / n_batches_train)

            prog_train.set_description(f"train, loss: {run.get_mavg_value('loss', 'train'):0.4f}, acc: {run.get_mavg_value('accuracy', 'train'):0.4f}")    
            if i % PLOT_STEPS == 0:
                run.plot()


        ds_val.reset()
        n_batches_val = BATCHES_PER_EPOCH if BATCHES_PER_EPOCH < len(ds_val) else len(ds_val)
        prog_val = tqdm(ds_val, total=n_batches_val, leave=False, desc='val')
        for i, (X, Y) in enumerate(prog_val):
            if i >= n_batches_val:
                break
            X = normalize(X)
            Y = one_hot_encode(Y, num_classes=NUM_CLASSES)
            Y_hat = model(X)

            loss = criterion.forward(Y_hat, Y)
            accuracy = compute_accuracy(Y_hat, Y)

            run.append('loss', 'val', loss, step_epoch=(epoch * n_batches_val + i) / n_batches_val)
            run.append('accuracy', 'val', accuracy, step_epoch=(epoch * n_batches_val + i) / n_batches_val)

            prog_val.set_description(f"val, loss: {run.get_mavg_value('loss', 'val'):0.4f}, acc: {run.get_mavg_value('accuracy', 'val'):0.4f}")
            if i % PLOT_STEPS == 0:
                run.plot()

        run.plot()
        model.save(f"{run.run_dir}/model.npz")

        loss_train_epoch = run.get_mavg_value('loss', 'train')
        accs_train_epoch = run.get_mavg_value('accuracy', 'train')
        loss_val_epoch = run.get_mavg_value('loss', 'val')
        acc_val_epoch = run.get_mavg_value('accuracy', 'val')
        print(f"Epoch {epoch}/{EPOCHS}, time {(datetime.now()-start).total_seconds()/60:.2f} min, loss_train {loss_train_epoch:0.4f}, loss_val {loss_val_epoch:0.4f}, acc_train {accs_train_epoch:0.4f}, acc_val {acc_val_epoch:0.4f}")
