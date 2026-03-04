import numpy as np
import os
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
    
class ConvLayer(Module):
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

        # create sliding windows of the input for convolution -> same as iterating over the input with nested loops, but much faster
        regions = sliding_window_view(
            input, 
            (self.kernel_size, self.kernel_size), 
            axis=(2, 3)     # over height and width dimensions
        )
        output = np.tensordot(
            regions,
            self.kernels,
            axes=(
                [1, 4, 5],  # over regions: in_channels, kernel_height, kernel_width
                [1, 2, 3]   # over kernels: in_channels, kernel_height, kernel_width
            )
        )
        output = np.transpose(output, (0, 3, 1, 2)) # rearrange to (batch_size, out_channels, output_height, output_width)

        return output
    
    def backward(self, grad_output:np.ndarray):
        _, _, height, width = self.input.shape
        regions = sliding_window_view(self.input, (self.kernel_size, self.kernel_size), axis=(2, 3))

        grad_kernels = np.tensordot(
            grad_output,
            regions,
            axes=(
                [0, 2, 3],  # over grad_output: batch_size, output_height, output_width
                [0, 2, 3]   # over regions: batch_size, output_height, output_width
            )
        )

        # To compute grad_input, we need to convolve the grad_output with the flipped kernels. We can use the same sliding window approach as in the forward pass,
        # but we need to pad the grad_output to account for the kernel size.
        pad = self.kernel_size - 1
        grad_output_padded = np.pad(grad_output, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        # Create sliding windows of the padded grad_output for convolution with flipped kernels
        grad_windows = sliding_window_view(grad_output_padded, (self.kernel_size, self.kernel_size), axis=(2, 3))
        kernels_flipped = np.flip(self.kernels, axis=(2, 3))
        grad_input = np.tensordot(
            grad_windows,
            kernels_flipped,
            axes=([1, 4, 5], [0, 2, 3])
        )
        grad_input = np.transpose(grad_input, (0, 3, 1, 2)) # rearrange to (batch_size, in_channels, height, width)

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

    def print_shapes(self, input:np.ndarray):
        print(f"{'Input':<15}shape: {input.shape[1:]}")
        module_names = {}
        for module in self.modules:
            input = module.forward(input)
            if module.__class__.__name__ not in module_names:
                module_names[module.__class__.__name__] = 0
            module_names[module.__class__.__name__] += 1
            module_name = f"{module.__class__.__name__}_{module_names[module.__class__.__name__]}"
            print(f"{module_name:<15}shape: {input.shape[1:]}")

