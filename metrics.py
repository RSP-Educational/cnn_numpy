import numpy as np

def compute_accuracy(predictions:np.ndarray, targets:np.ndarray):
    pred_labels = np.argmax(predictions, axis=1)
    target_labels = np.argmax(targets, axis=1)
    accuracy = np.mean(pred_labels == target_labels)
    return accuracy

class CrossEntropyLossWithSoftmax():
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