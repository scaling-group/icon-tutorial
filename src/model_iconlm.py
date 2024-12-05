import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from . import models_utils as mu  
from . import models_tran as mt
from . import data_utils as du
from omegaconf import OmegaConf  
import hydra
from omegaconf import DictConfig


class ICON_LM(nn.Module):
    def __init__(self, config, data_shape, shot_num_min):
        super(ICON_LM, self).__init__()
        self.config = config
        self.data_shape = data_shape
        self.shot_num_min = shot_num_min

        input_dim = data_shape.demo_cond_k[-1] + data_shape.demo_cond_v[-1]
        output_dim = data_shape.demo_qoi_v[-1]
        # for training
        basic_mask, index_pos, out_mask = mu.build_matrices(data_shape, mode='train', shot_num_min = shot_num_min)   
        self.register_buffer('basic_mask', basic_mask) # becomes a constant in the model
        self.register_buffer('index_pos', index_pos)
        self.register_buffer('out_mask', out_mask)

        fig = mu.plot_model_consts(basic_mask, index_pos[:,None], out_mask)
        plt.savefig("const_train.png")

        # for test, we cache the masks and index_pos for different data shapes
        self.basic_mask_test = {}
        self.index_pos_test = {}

        self.pre_projection = nn.Linear(input_dim, config['transformer']['model_dim'])
        if config['pe'] == 'learned':
            max_pos = self.index_pos.max().item() + 1
            self.func_pe = nn.Embedding(max_pos, config['transformer']['model_dim'])
        
        if config['tran'] == 'built-in':
            print("Using built-in transformer", flush=True)
            encoder_layer = nn.TransformerEncoderLayer(d_model= config['transformer']["model_dim"],
                                                    nhead= config['transformer']["n_heads"],
                                                    dim_feedforward= config['transformer']["model_dim"] * config['transformer']["widening_factor"],
                                                    dropout= 0.0,
                                                    activation='gelu',
                                                    batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['transformer']["n_layers"])
        elif config['tran'] == 'custom':
            print("Using custom transformer", flush=True)
            encoder_layer = mt.TransformerEncoderLayer(d_model= config['transformer']["model_dim"],
                                                    nhead= config['transformer']["n_heads"],
                                                    dim_feedforward= config['transformer']["model_dim"] * config['transformer']["widening_factor"],
                                                    dropout= 0.0,
                                                    ff = config['ff'])
            self.transformer = mt.TransformerEncoder(encoder_layer, num_layers=config['transformer']["n_layers"])
        else:
            raise ValueError(f"{config['tran']} tranformer not supported")
        
        self.post_projection = nn.Linear(config['transformer']['model_dim'], output_dim)

    def _basic_forward(self, data, mode, index_pos, basic_mask, shot_num_min = None, need_weights = False):
        '''
        basic forward of the model, with flexibility, no post-processing
        '''
        batch_size = data.demo_cond_k.shape[0]
        demo_num = data.demo_cond_k.shape[1]
        cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu.build_bool_sequence(demo_num, mode=mode, shot_num_min=shot_num_min)

        sequence = mu.build_data_sequence(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list) # [batchsize, total_len, dim]
        sequence = self.pre_projection(sequence) # [batchsize, total_len, model_dim]
        
        if self.config['pe'] == 'learned':
            sequence = sequence + self.func_pe(index_pos) # [batchsize, total_len, model_dim]
        elif self.config['pe'] == 'no':
            pass # no pe, see https://arxiv.org/pdf/2203.16634
        else:
            raise ValueError("pe should be 'learned' or 'no'")

        if self.config.data_mask:
            # data_mask is used when some data points are masked out. 
            # It is not tested in this code, i.e., only data_mask = None is tested
            data_mask = mu.build_data_mask(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list) # [batchsize, total_len]
            data_mask = ~data_mask # -> zero for "attention", one for "no attention"
        else:
            data_mask = None # all data is used

        # careful: here basic_mask use zero for "no attention" and one for "attention"
        if need_weights:
            sequence, weights = self.transformer(sequence, mask = ~basic_mask, src_key_padding_mask = data_mask, need_weights = True)
            sequence = self.post_projection(sequence) # [batchsize, total_len, out_dim]
            return sequence, weights
        else:
            sequence = self.transformer(sequence, mask = ~basic_mask, src_key_padding_mask = data_mask) # [batchsize, total_len, model_dim]
            sequence = self.post_projection(sequence) # [batchsize, total_len, out_dim]
            return sequence


    def _train_forward(self, data, reshape = True):
        '''
        training forward, predict demo_qoi_v from shot_num_min, and quest_qoi_v 
        '''
        sequence = self._basic_forward(data, 'train', self.index_pos, self.basic_mask, self.shot_num_min)
        sequence = sequence[:,self.out_mask,:]  # [batchsize, out_len, dim]
        if reshape:
            bs, _, qoi_len, dim = data.demo_qoi_v.shape
            sequence = sequence.view(bs, -1, qoi_len, dim) # [batchsize, x, qoi_len, dim], x depends on demo_num and shot_num_min
        return sequence

    def _test_forward(self, data, need_weights = False):
        '''
        predict quest_qoi_v (with all demos)
        '''
        # get tuple so that it can be used as key, exclude batch dim since it won't affect the mask etc.
        data_shape_nt = data.get_shape_namedtuple()
        if data_shape_nt not in self.basic_mask_test:
            # build masks and index_pos on the fly
            data_shape = data.get_shape()
            basic_mask, index_pos = mu.build_matrices(data_shape, mode='test', returns = ["mask", "index"])
            device = next(self.parameters()).device
            basic_mask = basic_mask.to(device)
            index_pos = index_pos.to(device)
            # save for next time
            self.basic_mask_test[data_shape_nt] = basic_mask
            self.index_pos_test[data_shape_nt] = index_pos
            print("saved new basic_mask and index_pos for shape", data_shape_nt)
        else:
            # datashape as training, using pre-built masks and index_pos
            basic_mask = self.basic_mask_test[data_shape_nt]
            index_pos = self.index_pos_test[data_shape_nt]

        if need_weights:
            sequence, weights = self._basic_forward(data, 'test', index_pos, basic_mask, need_weights = True)
            quest_qoi_len = data.quest_qoi_mask.shape[-1]
            sequence = sequence[:, -quest_qoi_len:, :]
            return sequence, weights
        else:
            sequence = self._basic_forward(data, 'test', index_pos, basic_mask)
            quest_qoi_len = data.quest_qoi_mask.shape[-1]
            sequence = sequence[:, -quest_qoi_len:, :]
            return sequence

    def forward(self, data, mode, **kwargs):
        if mode == 'train':
            return self._train_forward(data, **kwargs)
        elif mode == 'test':
            return self._test_forward(data, **kwargs)
        else:
            raise ValueError("mode should be 'train' or 'test'")



@hydra.main(version_base=None, config_path="../configs/", config_name="default")
def main(cfg: DictConfig):

    from . import utils
    utils.set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = cfg.model

    test_data, _ = du.build_dummy_data(batch_size=3, demo_num=5, seq_len=50, k_dim=1, v_dim=1)
    test_data = test_data.to(device)
    data_shape = test_data.get_shape()
    test_data.print_shape()


    basic_mask, index_pos, out_mask = mu.build_matrices(data_shape, mode='test', shot_num_min=0)
    fig = mu.plot_model_consts(basic_mask, index_pos[:,None], out_mask)
    plt.savefig("model_consts_test.png")

    basic_mask, index_pos, out_mask = mu.build_matrices(data_shape, mode='train', shot_num_min=0)
    fig = mu.plot_model_consts(basic_mask, index_pos[:,None], out_mask)
    plt.savefig("model_consts_train.png")

    demo_num = test_data.demo_cond_k.shape[1]
    cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu.build_bool_sequence(demo_num, mode="train", shot_num_min=0)

    sequence = mu.build_data_sequence(test_data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)
    print("Sequence shape:", sequence.shape) # torch.Size([3, 850, 2])

    mask = mu.build_data_mask(test_data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)
    print("Data Mask shape:", mask.shape) # torch.Size([3, 850])


    icon_lm_model = ICON_LM(model_config, data_shape, shot_num_min=0)
    icon_lm_model.to(device)


    train_forward = icon_lm_model(test_data, mode = 'train')
    print("train forward shape:", train_forward.shape)  # torch.Size([3, 6, 50, 1])

    quest_forward = icon_lm_model(test_data, mode = 'test')
    print("quest forward shape:", quest_forward.shape)  # torch.Size([3, 50, 1])

    flexible_data, _ = du.build_dummy_data(batch_size=3, demo_num=4, seq_len=20, k_dim=1, v_dim=1)
    flexible_data = flexible_data.to(device)
    flexible_forward = icon_lm_model(flexible_data, mode = 'test')
    print("flexible_predicted shape:", flexible_forward.shape)  # torch.Size([3, 20, 1])

    flexible_data, _ = du.build_dummy_data(batch_size=3, demo_num=0, seq_len=20, k_dim=1, v_dim=1)
    flexible_data = flexible_data.to(device)
    flexible_forward = icon_lm_model(flexible_data, mode = 'test')
    flexible_forward = icon_lm_model(flexible_data, mode = 'test') # check using cached masks
    print("flexible_predicted shape:", flexible_forward.shape)  # torch.Size([3, 20, 1]), even with no demos

    print(quest_forward[:,0,0])


if __name__ == "__main__":
    main()