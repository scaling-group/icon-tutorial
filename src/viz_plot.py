import matplotlib.pyplot as plt
from . import data_utils as du
from . import viz_utils as vu
import jax
from .datagen.weno.weno_solver import generate_weno_scalar_sol
import numpy as np
import torch

def get_x_from_feat(feature, kfeat):
    '''
    feature: [..., kdim]
    kfeat: string
    return: [..., 1]
    '''
    if kfeat == "x":
        x = feature
    elif kfeat == "sin":
        x = np.arctan2(feature[..., 1], feature[..., 0])[..., None]
        x = x / (2 * np.pi) + 0.5
    return x

def plot_data_pred_label(data, pred, label, kfeat, prefix = "", fig = None):
    '''
    assume data has no batch dim
    disregard the mask for now
    '''
    if type(data.demo_cond_k) == torch.Tensor:
        data = data.to_numpy()
    if type(pred) == torch.Tensor:
        pred = pred.numpy()
    if type(label) == torch.Tensor:
        label = label.numpy()
    
    demo_num = data.demo_cond_k.shape[0]
    if fig is None:
        fig = plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    for i in range(demo_num):
        plt.plot(get_x_from_feat(data.demo_cond_k[i,:,:], kfeat)[...,0], data.demo_cond_v[i,:,0], '*', alpha = 0.4)
    plt.plot(get_x_from_feat(data.quest_cond_k[0,:,:], kfeat)[...,0], data.quest_cond_v[0,:,0], 'bo')
    plt.title("COND")

    plt.subplot(2, 1, 2)
    for i in range(demo_num):
        plt.plot(get_x_from_feat(data.demo_qoi_k[i,:,:], kfeat)[...,0], data.demo_qoi_v[i,:,0], '*', alpha = 0.4)
    plt.plot(get_x_from_feat(data.quest_qoi_k[0,:,:], kfeat)[...,0], label[0,:,0], 'ko')
    plt.plot(get_x_from_feat(data.quest_qoi_k[0,:,:], kfeat)[...,0], pred[0,:,0], 'r+')
    plt.title("QOI")
    plt.suptitle(prefix + data.equation)
    plt.tight_layout()
    return fig

def plot_attn_weights(weights, patch_size, prefix = ""):
    '''
    weights: [num_layers, heads, seq_len, seq_len], assume no batch dim
    '''
    num_layers, heads, seq_len, _ = weights.shape
    fig = plt.figure(figsize= (5 * num_layers, 5 * heads))
    for i in range(num_layers):
        for j in range(heads):
            ax = plt.subplot(heads, num_layers, j * num_layers + i + 1)
            ax.imshow(weights[i, j, :, :], cmap='hot', interpolation="nearest")
            # Add gridlines at each patch boundary
            for x in range(patch_size, seq_len, patch_size):
                ax.axvline(x-0.5, color='white', linestyle='--', linewidth=1)  # vertical lines
                ax.axhline(x-0.5, color='white', linestyle='--', linewidth=1)  # horizontal lines
            plt.title(f"Layer {i}, Head {j}")
    plt.suptitle(prefix)
    plt.tight_layout()
    return fig


def plot_sequence(seq, splits):
    '''
    splits = demo_num + 1
    seq: (seq_len, 2) -> (splits, 3, seq_len, 2)
    '''
    split_seq = seq.view(splits, 3, -1, 2)
    fig = plt.figure(figsize=(12, 6))
    for k in range(3):
        plt.subplot(2, 3, k+1)
        for i in range(splits):
            plt.plot(split_seq[i,k,:,0])
    for k in range(3):
        plt.subplot(2, 3, k+4)
        for i in range(splits):
            plt.plot(split_seq[i,k,:,1])
    plt.tight_layout()
    return fig


def apply_operator(eqn, init, param = None):
    '''
    assume no batch dim
    eqn: string
    init: [num, seq_len, 1]
    param: (3,)
    return: [num, seq_len, 1]
    '''
    eqn_split = eqn.split('_')
    coeff_a, coeff_b, coeff_c, steps = [float(e) for e in eqn_split[-6:-2]]
    if param is not None: # overwrite the eqn
        coeff_a, coeff_b, coeff_c = [float(e) for e in param]
    steps = int(steps)
    fn = jax.jit(lambda u: coeff_a * u * u * u + coeff_b * u * u + coeff_c * u)
    grad_fn = jax.jit(lambda u: 3 * coeff_a * u * u + 2 * coeff_b * u + coeff_c)
    sol = generate_weno_scalar_sol(dx=0.01, dt=0.0005, init = init, fn=fn, steps=steps, grad_fn=grad_fn, stable_tol=None)  # (num, steps + 1, N, 1)
    term = sol[:, -1, :, :]
    return np.array(term)


# Usage example
if __name__ == "__main__":
    import glob
    from . import models_utils as mu

    data_files = "./scratch/data/data_weno_cubic/train*forward*.h5"  # HDF5 files 
    batch_size = 16
    num_workers = 4
    demo_num = 5
    cfg = {"demo_num": demo_num, "kfeat": "sin"}

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    weights = torch.randn(3, 4, 400, 400)
    fig = plot_attn_weights(weights, 100)
    fig.savefig("attn_weights.png")

    file_paths = glob.glob(data_files) # Get a list of all files matching the pattern
    dataset = du.WenoDataset(file_paths, cfg, "train")

    dataloader = du.DataLooper(dataset, batch_size, num_workers, infinite=True)
    for i in range(10000):
        print(i, end = ",", flush=True)
        data, label = next(dataloader)
        data.print_shape()
        pred = label + 0.1 * torch.randn_like(label)

        figs = []
        pred, label = pred[0,...], label[0,...]

        for i in [0, demo_num]:
            this_data = data.get_n_demos(i)
            this_data = this_data.get_one_batch(0)
            fig = plot_data_pred_label(this_data, pred, label, kfeat = cfg["kfeat"])
            fig.savefig(f"demo_pred_label.png")
            figs.append(fig)    
            # Apply the operator to reconstruct the qoi
            recons_demo_qoi_v = apply_operator(this_data.equation, this_data.demo_cond_v.numpy(), this_data.param)
            recons_label = apply_operator(this_data.equation, this_data.quest_cond_v.numpy(), this_data.param)
            if i == demo_num:
                # confirm the reconstruction is the same as the original data
                print(np.mean((this_data.demo_qoi_v.numpy() - recons_demo_qoi_v)**2))
                print(np.mean((label.numpy()- recons_label)**2))
            this_data.demo_qoi_v = torch.tensor(recons_demo_qoi_v)
            fig = plot_data_pred_label(this_data, pred, recons_label, kfeat = cfg["kfeat"], fig = fig)
            fig.savefig(f"demo_pred_label_reconst.png")
        
        final_fig = vu.merge_images([figs]) # PIL image
        final_fig.save(f"demo_pred_label_all.png")

        cond_bool_list = [True] * demo_num + [True]
        qoi_kv_bool_list = [True] * demo_num + [True]
        qoi_k_bool_list = [True] * demo_num + [True]
        seq = mu.build_data_sequence(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)
        fig = plot_sequence(seq[0,...], splits = demo_num+1)
        fig.savefig(f"sequence.png")
        if i >= 0:
            break

