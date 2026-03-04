import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from pathlib import Path
from data import MNISTDataset, normalize
from model import Module, DenseLayer, Flatten, ConvLayer, ReLU, MaxPool, NeuralNetwork

def plot_images(images:np.ndarray, labels:np.ndarray, num_images:int = 8, n_cols:int = 8):
    """Plot a grid of images with their corresponding labels.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, H, W) containing the images to plot.

    labels : np.ndarray
        Array of shape (N,) containing the labels corresponding to the images.

    num_images : int
        Number of images to plot. Default is 10.
    """
    if images.shape[1] == 1:
        images_vis = np.squeeze(images.copy(), axis=1)
        
    plt.figure(figsize=(24, 24))
    for i in range(len(images_vis)):
        if i >= num_images:
            break
        plt.subplot((num_images + n_cols - 1) // n_cols, n_cols, i + 1)
        cv.rectangle(images_vis[i], (0, 0), (images_vis[i].shape[1]-1, images_vis[i].shape[0]-1), (0.5, 0.5, 0.5), 2)
        plt.imshow(images_vis[i], cmap='gray')
        plt.title(f'{labels[i]}')
        plt.axis('off')
    plt.show()

def visualize_features(image:np.ndarray, model:NeuralNetwork, n_kernels:int = 5, zscore_input:bool = False, show_stats:bool = True):
    def toImage(array:np.ndarray, target_size:int, margin:int, min:int = 0, max:int = 255, cmap_name:str | None = None, normalized_input:bool = False):
        if normalized_input:
            normalized = np.clip(array, 0.0, 1.0)
        else:
            array_min, array_max = np.percentile(array, [2, 98])
            denom = (array_max - array_min) if (array_max - array_min) > 1e-12 else 1.0
            normalized = np.clip((array - array_min) / denom, 0.0, 1.0)

        if cmap_name is not None:
            cmap = plt.get_cmap(cmap_name)
            img_rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
            #img = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
            img = img_rgb
        else:
            rescaled = normalized * (max - min) + min
            img = rescaled.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        img = cv.resize(img, (target_size-2*margin, target_size-2*margin), interpolation=cv.INTER_NEAREST)
        img_up = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
        img_up[margin:margin+img.shape[0], margin:margin+img.shape[1], :] = img
        return img_up

    def draw_centered_max_value(img_fmap:np.ndarray, fmap_max:float):
        txt = f'{fmap_max:.3f}'
        txt_scale_large = 1.0
        txt_thickness_large = 2
        (tw, th), _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, txt_scale_large, txt_thickness_large)
        tx = (img_fmap.shape[1] - tw) // 2
        ty = (img_fmap.shape[0] + th) // 2

        x1 = max(tx, 0)
        y1 = max(ty - th, 0)
        x2 = min(tx + tw, img_fmap.shape[1])
        y2 = min(ty, img_fmap.shape[0])
        region = img_fmap[y1:y2, x1:x2]

        mean_luma = float(np.mean(region)) if region.size > 0 else 127.0
        txt_color = (0, 0, 0) if mean_luma > 140.0 else (255, 255, 255)

        cv.putText(img_fmap, txt, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, txt_scale_large, txt_color, txt_thickness_large)

    txt_scale = 0.9
    txt_thickness = 4

    image_proc = image.copy().astype(np.float32)
    if np.max(image_proc) > 1.0:
        image_proc = image_proc / 255.0
    if zscore_input:
        image_proc = normalize(image_proc)

    X = np.expand_dims(image_proc, (0, 1))

    img_size = 300
    img_kernel_size = int(round(0.2 * img_size))
    margin = 10
    inner_margin = 5

    conv_idx, pool_idx = 0, 0
    visual_layer_indices = [
        idx for idx, module in enumerate(model.modules)
        if type(module).__name__ in ['ConvLayer', 'MaxPool']
    ]
    last_visual_layer_idx = visual_layer_indices[-1] if len(visual_layer_indices) > 0 else -1

    image = toImage(image, img_size, margin=margin)
    img_up = cv.resize(image, (img_size, img_size), interpolation=cv.INTER_NEAREST)
    cv.rectangle(img_up, (3, 3), (img_up.shape[1]-4, img_up.shape[0]-4), (180, 180, 180), 4)
    for i in range(n_kernels - 1):
        img_up = cv.vconcat([img_up, np.full((img_size, img_size, 3), 255, dtype=np.uint8)])

    output = X
    for module_idx, module in enumerate(model.modules):
        output = module.forward(output)
        if type(module).__name__ == 'ConvLayer':
            # WICHTIG: gemeinsame Skala pro Layer (nicht pro Kanal),
            # damit schwache Feature-Maps nicht künstlich kontrastreich wirken.
            max_abs = np.percentile(np.abs(output[0]), 99)
            if max_abs < 1e-12:
                max_abs = 1.0
            output_scaled = np.clip(0.5 + 0.5 * output / max_abs, 0, 1)
        else:
            o_min, o_max = np.percentile(output, [1, 99])
            denom = (o_max - o_min) if (o_max - o_min) > 1e-12 else 1e-12
            output_scaled = np.clip((output.copy() - o_min) / denom, 0, 1)

        if type(module).__name__ in ['ConvLayer', 'MaxPool']:
            is_last_visual_layer = (module_idx == last_visual_layer_idx)
            color_module = (255, 0, 0) if type(module).__name__ == 'ConvLayer' else (0, 0, 255)

            img_module = np.full((img_size * n_kernels, img_size, 3), 255, dtype=np.uint8)

            if type(module).__name__ == 'ConvLayer':
                k_min, k_max = np.percentile(module.kernels, [1, 99])
                k_denom = (k_max - k_min) if (k_max - k_min) > 1e-12 else 1.0
                for i in range(n_kernels):
                    if i >= len(module.kernels):
                        break
                    feature_map = output_scaled[0, i].copy()
                    img_fmap = toImage(feature_map, img_size, margin=margin, cmap_name='RdBu_r', normalized_input=True)

                    if show_stats and is_last_visual_layer:
                        fmap_raw = output[0, i]
                        fmap_max = float(np.max(fmap_raw))
                        draw_centered_max_value(img_fmap, fmap_max)

                    kernel = np.clip((module.kernels[i, 0].copy() - k_min) / k_denom, 0, 1)
                    img_kernel = toImage(kernel, img_kernel_size, margin=0, cmap_name='RdBu_r', normalized_input=True)
                    cv.rectangle(img_kernel, (0, 0), (img_kernel.shape[1]-1, img_kernel.shape[0]-1), color_module, 3)

                    img_fmap[0:img_kernel.shape[0], 0:img_kernel.shape[1]] = img_kernel
                    
                    sy = i * img_size
                    ey = sy + img_size
                    img_module[sy:ey, :] = img_fmap
                (w, h), _ = cv.getTextSize(f'Conv Layer {conv_idx + 1}', cv.FONT_HERSHEY_SIMPLEX, txt_scale, txt_thickness)
                cv.rectangle(img_module, (img_size//2 - w//2 - inner_margin, img_module.shape[0] - margin - h - 4*inner_margin), (img_size//2 + w//2 + inner_margin, img_module.shape[0] - margin - inner_margin), (255, 255, 255), cv.FILLED)
                cv.putText(img_module, f'Conv Layer {conv_idx + 1}', (img_size//2-w//2, img_module.shape[0] - margin  - 3*inner_margin), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color_module, txt_thickness)

                conv_idx += 1
            elif type(module).__name__ == 'MaxPool':
                for i in range(n_kernels):
                    if i >= output_scaled.shape[1]:
                        break
                    feature_map = output_scaled[0, i].copy()
                    img_fmap = toImage(feature_map, img_size, margin=margin, cmap_name='RdBu_r', normalized_input=True)

                    if show_stats and is_last_visual_layer:
                        fmap_raw = output[0, i]
                        fmap_max = float(np.max(fmap_raw))
                        draw_centered_max_value(img_fmap, fmap_max)

                    sy = i * img_size
                    ey = sy + img_size
                    img_module[sy:ey, :] = img_fmap
                (w, h), _ = cv.getTextSize(f'MaxPool Layer {pool_idx + 1}', cv.FONT_HERSHEY_SIMPLEX, txt_scale, txt_thickness)
                cv.rectangle(img_module, (img_size//2 - w//2 - inner_margin, img_module.shape[0] - margin - h - 4*inner_margin), (img_size//2 + w//2 + inner_margin, img_module.shape[0] - margin - inner_margin), (255, 255, 255), cv.FILLED)
                cv.putText(img_module, f'MaxPool Layer {pool_idx + 1}', (img_size//2-w//2, img_module.shape[0] - margin  - 3*inner_margin), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color_module, txt_thickness)

                pool_idx += 1

            cv.rectangle(img_module, (3, 3), (img_module.shape[1]-4, img_module.shape[0]-4), color_module, 4)
            img_up = cv.hconcat([img_up, img_module])


    fig, (ax_img, ax_cbar) = plt.subplots(
        1,
        2,
        figsize=(5, 10),
        gridspec_kw={'width_ratios': [30, 1], 'wspace': 0.05}
    )

    ax_img.imshow(img_up)
    ax_img.axis('off')

    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Aktivierung (relativ)')
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    plt.show()
    plt.close(fig)
    # cv.imshow('Feature Visualization', img_up)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def visualize_first_layer_kernels(model: NeuralNetwork, max_kernels: int = 8, channel: int = 0):
    first_conv = None
    for module in model.modules:
        if hasattr(module, 'kernels') and isinstance(module.kernels, np.ndarray) and module.kernels.ndim == 4:
            first_conv = module
            break

    if first_conv is None:
        raise ValueError('Kein ConvLayer mit kernels im Modell gefunden.')

    kernels = first_conv.kernels
    if channel < 0 or channel >= kernels.shape[1]:
        raise ValueError(f'channel muss zwischen 0 und {kernels.shape[1]-1} liegen.')

    kernels_vis = kernels[:, channel]
    n = min(max_kernels, kernels_vis.shape[0])
    n_cols = min(8, n)
    n_rows = int(np.ceil(n / n_cols))

    abs_max = np.max(np.abs(kernels_vis[:n]))
    if abs_max < 1e-12:
        abs_max = 1.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(n_rows, n_cols)

    for idx in range(n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        if idx < n:
            im = ax.imshow(kernels_vis[idx], cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
            ax.set_title(f'K{idx+1}', fontsize=9)
        ax.axis('off')

    fig.suptitle('First Conv Layer Kernels (zentriert, +/-)', fontsize=12)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    RUN_ID              = 'ConvModel-v1'
    ds_train = MNISTDataset(split = 'train', batch_size=128)
    model = NeuralNetwork(modules=[
        ConvLayer(in_channels=1, out_channels=32, kernel_size=3),
        ReLU(),
        MaxPool(kernel_size=2, stride=2),

        ConvLayer(in_channels=32, out_channels=64, kernel_size=3),
        ReLU(),
        MaxPool(kernel_size=2, stride=2),

        Flatten(),

        DenseLayer(in_features=64*5*5, out_features=10),
    ])
    model.load(f'runs/{RUN_ID}/model.npz')
    visualize_features(ds_train[0][0], model)