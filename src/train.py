import wandb
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint
import numpy as np
import pytz
from datetime import datetime
import glob

from . import utils
from . import model_iconlm
from . import data_utils as du
from . import viz_plot as vp
from . import viz_utils as vu
from . import trainers


def to_save(step, cfg):
    return (step % cfg.save_interval == 0) and (step > 0)

def to_print(step, cfg):
    to_print_1 = (step < cfg.print_interval) and (step % (cfg.print_interval//10) == 0)
    to_print_2 = (step % cfg.print_interval == 0)
    return to_print_1 or to_print_2

def to_time(step, cfg):
    to_time_0 = (step > cfg.time_warm_up)
    to_time_1 = (step == cfg.time_warm_up + 100)
    to_time_2 = (step % cfg.time_interval == 0)
    to_time_3 = (step < cfg.time_interval) and (step % (cfg.time_interval//10) == 0)
    return to_time_0 and (to_time_1 or to_time_2 or to_time_3)

def to_plot(step, cfg):
    to_plot_1 = (step % cfg.plot_interval == 0)
    to_plot_2 = (step == 1) # plot the initial state
    return (to_plot_1 or to_plot_2)

def run_train(cfg):

    if cfg.board:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    stamp = utils.timer.get_time_stamp()
    print("stamp: {}".format(stamp))

    print(OmegaConf.to_yaml(cfg))

    train_dataset = du.WenoDataset(glob.glob(cfg.data.train_data_files), cfg.data, "train")
    train_datalooper = du.DataLooper(train_dataset, cfg.data.batch_size, cfg.data.num_workers, infinite=True)

    valid_dataset = du.WenoDataset(glob.glob(cfg.data.valid_data_files), cfg.data, "valid")
    valid_datalooper = du.DataLooper(valid_dataset, cfg.data.batch_size, cfg.data.num_workers, infinite=True)

    this_data, this_label = next(valid_datalooper)
    model = model_iconlm.ICON_LM(cfg.model, this_data.get_shape(), cfg.loss.shot_num_min)
    trainer = trainers.Trainer(model, cfg, device, print_model = True)


    for _ in range(cfg.total_steps + 1):

        this_data, this_label = next(train_datalooper)
        trainer.iterate(this_data, this_label)

        if to_save(trainer.step, cfg):
            save_dir = os.path.join(cfg.save_path, stamp)
            os.makedirs(save_dir, exist_ok=True)
            trainer.save(save_dir)
        
        if to_print(trainer.step, cfg):
            train_data, train_label = this_data, this_label
            train_loss = trainer.get_loss(train_data, train_label)
            valid_data, valid_label = next(valid_datalooper)
            valid_loss = trainer.get_loss(valid_data, valid_label)
            utils.logger.info(f"step: {trainer.step}, train loss: {train_loss}, valid loss: {valid_loss}")            
            log_dict = {"train_loss": train_loss, "valid_loss": valid_loss}
            for demos in range(cfg.data.demo_num+1): # use part of demos
                slice_valid_data = valid_data.get_n_demos(demos)
                slice_valid_error = trainer.get_error(slice_valid_data, valid_label)
                print(f"step: {trainer.step}, demo_num: {demos}, valid error: {slice_valid_error}", flush=True)
                log_dict[f"demo_num_{demos}_valid_error"] = slice_valid_error
            if cfg.board: wandb.log({"step": trainer.step, **log_dict})

        if to_plot(trainer.step, cfg):
            bid = 0
            valid_data, valid_label = next(valid_datalooper)
            valid_data = valid_data.get_one_batch(bid, keep_dim = True)
            valid_label = valid_label[bid:bid+1,...] # only plot one batch
            plot_dict = {}
            for demos in range(cfg.data.demo_num+1): # use part of demos
                slice_valid_data = valid_data.get_n_demos(demos)
                slice_valid_pred = trainer.get_pred(slice_valid_data) # already numpy
                pred_fig = vp.plot_data_pred_label(slice_valid_data.get_one_batch(0), slice_valid_pred[0,...], 
                                                valid_label[0,...], kfeat = cfg.data.kfeat, prefix = f"step {trainer.step}: ")
                plot_dict[f"demo_num_{demos}"] = vu.fig_to_wandb(pred_fig)
            if cfg.board: wandb.log({"step": trainer.step, **plot_dict})

        if trainer.step == cfg.time_warm_up: # exclude warming up steps
            utils.timer.tic("time estimate")
        
        if to_time(trainer.step, cfg):
            ratio = (trainer.step - cfg.time_warm_up)/cfg.total_steps
            samples_processed = (trainer.step - cfg.time_warm_up) * cfg.data.batch_size
            utils.timer.estimate_time("time estimate", ratio, samples_processed)

    if cfg.board:
        wandb.finish()


@hydra.main(version_base=None, config_path="../configs/", config_name="default")
def main(cfg: DictConfig):
    run_train(cfg)


if __name__ == "__main__":
    main()