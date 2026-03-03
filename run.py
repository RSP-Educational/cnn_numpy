import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = {
    'train': 'blue',
    'val': 'orange',
    'test': 'green'
}

class Run():
    def __init__(self, run_id:str, mavg_window:int = 100):
        self.run_id = run_id
        self.data = {}
        self.run_dir = Path('runs') / self.run_id
        self.plot_dir = self.run_dir / 'plot'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.mavg_window = mavg_window

    def append(self, key:str, split:str, value:float, step_epoch:float):
        if key not in self.data:
            self.data[key] = {}
        if split not in self.data[key]:
            self.data[key][split] = {
                'val': ([], []),
                'avg': ([], [])
            }

        epoch = len(self.data[key][split]['val'][0]) / self.mavg_window
        self.data[key][split]['val'][0].append(step_epoch)
        self.data[key][split]['val'][1].append(value)

        if len(self.data[key][split]['val'][0]) >= self.mavg_window:
            mavg = np.mean(self.data[key][split]['val'][1][-self.mavg_window:])
            self.data[key][split]['avg'][0].append(step_epoch)
            self.data[key][split]['avg'][1].append(mavg)

    def plot(self):
        for key in self.data:
            plot_file = self.plot_dir / f'{key}.png'
            plt.figure(figsize=(10, 7))

            xmax = 0.0001
            ymax = 0.1
            for split in self.data[key]:
                color = COLORS[split] if split in COLORS else 'grey'
                epochs = self.data[key][split]['val'][0]
                values = self.data[key][split]['val'][1]
                mavg_epochs = self.data[key][split]['avg'][0]
                mavg_values = self.data[key][split]['avg'][1]
                plt.plot(epochs, values, alpha=0.3, color=color)
                plt.plot(mavg_epochs, mavg_values, label=split, color=color)
                xmax = max(xmax, epochs[-1])
                ymax = max(ymax, 1.05 * np.max(values))
                ymax = min(1.0, ymax)
            plt.xlim(0, xmax)
            plt.ylim(0, ymax)
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.6)
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.4)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.title(key)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()

    def get_value(self, key:str, split:str):
        if key in self.data and split in self.data[key]:
            return self.data[key][split]['val'][1][-1]
        else:
            return np.nan
        
    def get_mavg_value(self, key:str, split:str):
        if key in self.data and split in self.data[key]:
            if len(self.data[key][split]['avg'][0]) == 0:
                return np.mean(self.data[key][split]['val'][1])
            return self.data[key][split]['avg'][1][-1]
        else:
            return np.nan