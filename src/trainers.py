import torch
import torch.optim as optim
import torch.nn as nn
from .model_iconlm import ICON_LM
from . import utils
from . import model_iconlm  
from . import data_utils as du  
from tabulate import tabulate  
from omegaconf import OmegaConf  
import hydra
from omegaconf import DictConfig
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np


class WarmupCosineDecayScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
          lr_factor = epoch * 1.0 / self.warmup
        else:
          progress = (epoch - self.warmup) / (self.max_num_iters - self.warmup)
          lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return lr_factor
    

class Trainer:
    def __init__(self, model, cfg, device, print_model=False):

        self.cfg = cfg
        self.device = device

        if cfg.acc.compile:
            try:
                print("Compiling the model with torch.compile...", flush=True)
                self.model = torch.compile(model)  # Compile the model, but seems not helping in acceleration
            except AttributeError:
                print("torch.compile is not available. Using the model as-is.", flush=True)
                self.model = model
        else:
            print("Not compiling the model.", flush=True)
            self.model = model

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
            print("Model wrapped by DataParallel", flush=True)

        self.model.to(self.device)
        print("Model moved to {}".format(self.device), flush=True)

        model_single = self.model.module if hasattr(self.model, "module") else self.model

        headers = ["Parameter Name", "Shape", "Requires Grad"]
        table_data = [(name, str(param.shape), param.requires_grad) for name, param in model_single.named_parameters()]
        if print_model: print(tabulate(table_data, headers=headers, tablefmt="grid"))

        total_params = sum(p.numel() for p in model_single.parameters())
        trainable_params = sum(p.numel() for p in model_single.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model_single.parameters()),
            lr=self.cfg.opt.peak_lr,
            weight_decay=self.cfg.opt.weight_decay,
        )

        self.lr_scheduler = WarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup=int(self.cfg.opt.warmup_percent * self.cfg.total_steps // 100),
            max_iters=int(self.cfg.opt.decay_percent * self.cfg.total_steps // 100),
        )
        
        amp_dtype_dict = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        self.amp_dtype = amp_dtype_dict[self.cfg.acc.amp_dtype]
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.acc.amp)
        print("Using automatic mixed precision:", self.cfg.acc.amp, self.amp_dtype, flush=True)

        if self.cfg.acc.sdpa_cudnn:
            print("Using CUDNN for SDP attention", flush=True)
            self.sdpa_backends = [SDPBackend.CUDNN_ATTENTION]
        else:
            print("Using all available backends for SDP attention", flush=True)
            self.sdpa_backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, 
                                  SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]
        
        self.step = 0

    def _move_to_device(self, *tensor):
        if len(tensor) == 1:
            return tensor[0].to(self.device)
        return [t.to(self.device) for t in tensor]


    def _model_forward(self, data, mode, **kwargs):
        with sdpa_kernel(self.sdpa_backends):
            return self.model(data, mode, **kwargs)


    def _loss_fn(self, data, label):
        data, label = self._move_to_device(data, label)
        train_label = self._build_train_label(data, label)
        with torch.amp.autocast("cuda", enabled=self.cfg.acc.amp, dtype=self.amp_dtype):
            outputs = self._model_forward(data, "train")
            assert outputs.shape == train_label.shape
            loss = nn.MSELoss()(outputs, train_label)  # disregard the mask for now
        return loss

    def _build_train_label(self, data, label):
        demo_qoi_v = data.demo_qoi_v[:, self.cfg.loss.shot_num_min:, :, :]
        train_label = torch.cat([demo_qoi_v, label], dim=1) 
        return train_label

    def iterate(self, data, label):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._loss_fn(data, label)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        model_single = self.model.module if hasattr(self.model, "module") else self.model
        torch.nn.utils.clip_grad_norm_(model_single.parameters(), self.cfg.opt.gnorm_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()
        self.step += 1
        return loss.item()

    def save(self, save_dir):
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), "{}/{}_params.pth".format(save_dir, self.step))
        utils.logger.info("saved to {}, step {}".format(save_dir, self.step))

    def restore(self, save_dir, step, restore_opt_state=True):
        params_path = "{}/{}_params.pth".format(save_dir, step)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(torch.load(params_path, map_location=self.device))
        utils.logger.info("restored params from {}, step {}".format(save_dir, step))

    def get_loss(self, data, label):
        self.model.eval()
        with torch.no_grad():
            loss = self._loss_fn(data, label)
        return loss.item()

    def get_pred(self, data):
        '''
        get the prediction of quest_qoi_v
        '''
        data = self._move_to_device(data)
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.cfg.acc.amp, dtype=self.amp_dtype):
            output = self._model_forward(data, "test")
            output = output.unsqueeze(1)  # add the num dim in consistent with label
        return output.float().detach().cpu().numpy()

    def get_error(self, data, label):
        pred = self.get_pred(data)
        label = label.numpy()
        return np.sqrt(np.mean((pred - label) ** 2))

    def get_weights(self, data):
        data = self._move_to_device(data)
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.cfg.acc.amp, dtype=self.amp_dtype):
            _, weights = self._model_forward(data, "test", need_weights=True)
        weights = torch.stack(weights, dim=0) # (num_layers, batch_size, num_heads, seq_len, seq_len)
        weights = torch.transpose(weights, 0, 1)  # (batch_size, num_layers, num_heads, seq_len, seq_len)
        weights = weights.float().detach().cpu().numpy()
        return weights

@hydra.main(version_base=None, config_path="../configs/", config_name="default")
def main(cfg: DictConfig):
    utils.set_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_data, label = du.build_dummy_data(
        batch_size=cfg.data.batch_size,  
        demo_num=cfg.data.demo_num,      
        seq_len=50,  
        k_dim=1,           
        v_dim=1)


    data_shape = test_data.get_shape()
    model = ICON_LM(cfg.model, data_shape, cfg.loss.shot_num_min)
    trainer = Trainer(model, cfg, device)

    loss = trainer.iterate(test_data, label)
    print(f"Training loss: {loss}")

    valid_loss = trainer.get_loss(test_data, label)
    print(f"Validation loss: {valid_loss}")

    test_pred = trainer.get_pred(test_data)
    print(f"Test prediction: {test_pred.shape}")
    assert test_pred.shape == (cfg.data.batch_size, 1, 50, 1)

if __name__ == "__main__":
    main()

    