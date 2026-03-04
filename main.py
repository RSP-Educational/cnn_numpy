import numpy as np
import visualization as vis
from model import Module, NeuralNetwork, DenseLayer, Flatten, ConvLayer, MaxPool, ReLU
from datetime import datetime
from run import Run
from data import MNISTDataset, one_hot_encode, compute_accuracy, normalize
from tqdm import tqdm

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
    EPOCHS              = 15
    LEARNING_RATE       = 1e-3
    BATCH_SIZE          = 128
    NUM_CLASSES         = 10
    PLOT_STEPS          = 100
    BATCHES_PER_EPOCH   = 500000

    ds_train = MNISTDataset(split = 'train', batch_size=BATCH_SIZE)
    ds_val = MNISTDataset(split = 'val', batch_size=BATCH_SIZE)

    run = Run(run_id=RUN_ID, mavg_window=len(ds_train))
    #vis.plot_images(ds_train.images, ds_train.labels, num_images=40)

    model = NeuralNetwork(modules=[
        ConvLayer(in_channels=1, out_channels=32, kernel_size=3),
        ReLU(),
        MaxPool(kernel_size=2, stride=2),

        ConvLayer(in_channels=32, out_channels=64, kernel_size=3),
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
        run.mavg_window = n_batches_train
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
        run.mavg_window = n_batches_val
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
        print(f"Epoch {epoch+1}/{EPOCHS}, time {(datetime.now()-start)}, loss_train {loss_train_epoch:0.4f}, loss_val {loss_val_epoch:0.4f}, acc_train {accs_train_epoch:0.4f}, acc_val {acc_val_epoch:0.4f}")
