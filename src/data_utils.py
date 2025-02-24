import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
from pprint import pprint
from collections import namedtuple
import numpy as np
from . import utils

def dict_to_namedtuple(d, name="Data"):
    """ Convert a dictionary to a namedtuple recursively. """
    # Recursively apply to dictionaries within the dictionary
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namedtuple(value, name=key.capitalize())
    # Create the namedtuple type and instantiate it
    TupleType = namedtuple(name, d.keys())
    return TupleType(**d)


class Data:
    def __init__(self, equation = None, param = None,
                 demo_cond_k = None, demo_cond_v = None, demo_cond_mask = None,
                 demo_qoi_k = None, demo_qoi_v = None, demo_qoi_mask = None,
                 quest_cond_k = None, quest_cond_v = None, quest_cond_mask = None,
                 quest_qoi_k = None, quest_qoi_mask = None):
        self.equation = equation # tuple of strings
        self.param = param # torch.tensor
        self.demo_cond_k = demo_cond_k # torch.tensor
        self.demo_cond_v = demo_cond_v
        self.demo_cond_mask = demo_cond_mask
        self.demo_qoi_k = demo_qoi_k
        self.demo_qoi_v = demo_qoi_v
        self.demo_qoi_mask = demo_qoi_mask
        self.quest_cond_k = quest_cond_k
        self.quest_cond_v = quest_cond_v
        self.quest_cond_mask = quest_cond_mask
        self.quest_qoi_k = quest_qoi_k
        self.quest_qoi_mask = quest_qoi_mask

    def to(self, device):
        # Iterates over all attributes of the instance, moving each to the specified device
        # create a new object so the original data is not changed
        new_data = Data(equation = self.equation)
        for attr, value in self.__dict__.items():
            if attr != "equation":
                setattr(new_data, attr, value.to(device))
        return new_data

    def to_numpy(self):
        new_data = Data(equation = self.equation)
        for attr, value in self.__dict__.items():
            if attr != "equation":
                setattr(new_data, attr, value.numpy())
        return new_data

    def get_one_batch(self, bid, keep_dim = False):
        '''
        return new data with bid-th batch, can keep batch dim or not
        '''
        new_data = Data()
        for attr, value in self.__dict__.items():
            new_value = value[bid:bid+1] if keep_dim else value[bid]
            setattr(new_data, attr, new_value)
        return new_data

    def get_n_demos(self, demos):
        '''
        return new data with first n demos, may have batch dim or not
        '''
        has_batch = len(self.demo_cond_mask.shape) == 3
        new_data = Data()
        for attr, value in self.__dict__.items():
            if "demo" in attr:
                if has_batch:
                    setattr(new_data, attr, value[:, :demos, ...])
                else:
                    setattr(new_data, attr, value[:demos, ...])
            else:
                setattr(new_data, attr, value) # no change
        return new_data

    def print_shape(self):
        for attr, value in self.__dict__.items():
            if attr == "equation":
                if value is not None:
                    print(f"{attr} length:", len(value))
                else:
                    print(f"{attr} length: None")
            else:
                print(f"{attr}:", value.shape, value.dtype)

    def get_shape(self):
        data_shape = Data()
        for attr, value in self.__dict__.items():
            if attr == "equation":
                if value is not None:
                    setattr(data_shape, attr, len(value))
                else:
                    setattr(data_shape, attr, None)
            else:
                setattr(data_shape, attr, value.shape)
        return data_shape
    
    def get_shape_namedtuple(self, exclude_batch = True):
        data_shape = {}
        for attr, value in self.__dict__.items():
            if attr not in ["equation", "param"]:
                if exclude_batch:
                    data_shape[attr] = tuple(value.shape[1:])
                else:
                    data_shape[attr] = tuple(value.shape)
        return dict_to_namedtuple(data_shape)


def build_dummy_data(batch_size, demo_num, seq_len, k_dim, v_dim):
    data = Data(
        equation = ("dummy",)*batch_size,
        param = torch.randn((batch_size, 3)),
        demo_cond_k=torch.randn((batch_size, demo_num, seq_len, k_dim)),
        demo_cond_v=torch.randn((batch_size, demo_num, seq_len, v_dim)),
        demo_cond_mask=torch.ones((batch_size, demo_num, seq_len), dtype=torch.bool),
        demo_qoi_k=torch.randn((batch_size, demo_num, seq_len, k_dim)),
        demo_qoi_v=torch.randn((batch_size, demo_num, seq_len, v_dim)),
        demo_qoi_mask=torch.ones((batch_size, demo_num, seq_len), dtype=torch.bool),
        quest_cond_k=torch.randn((batch_size, 1, seq_len, k_dim)),
        quest_cond_v=torch.randn((batch_size, 1, seq_len, v_dim)),
        quest_cond_mask=torch.ones((batch_size, 1, seq_len), dtype=torch.bool),
        quest_qoi_k=torch.randn((batch_size, 1, seq_len, k_dim)),
        quest_qoi_mask=torch.ones((batch_size, 1, seq_len), dtype=torch.bool)
    )
    label = torch.randn((batch_size, 1, seq_len, v_dim))
    return data, label

class WenoDataset(Dataset):
    def __init__(self, file_paths, cfg, name = "WenoDataset"):
        self.file_paths = file_paths
        self.cfg = cfg
        self.name = name
        self.indices = []

        print('='*50)
        print(f"{name} loading {len(file_paths)} files:")
        pprint(file_paths)
        print('='*50, flush=True)

        self.worker_rng = None  # Placeholder for worker-specific random number generator
        self.worker_id = None  # Placeholder for worker ID
        self.file_handles = None # placeholder for file handles

        # Collecting a list of (file_path, group_name) for record indexing
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as f:
                self.indices.extend([(file_path, key) for key in f.keys()])
        print(f"{self.name} total number of records: {len(self.indices)}")

    def set_worker_rng(self, worker_id, worker_rng):
        self.worker_id = worker_id
        self.worker_rng = worker_rng
        self.file_handles = {}
        print(self.name, "Worker ID:", self.worker_id, "Worker Initial RNG Seed:", self.worker_rng.initial_seed(), flush=True)
        # make sure the worker_rng is unique for each worker

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_path, group_name = self.indices[idx]

        if file_path not in self.file_handles:
            self.file_handles[file_path] = h5py.File(file_path, 'r')

        group = self.file_handles[file_path][group_name]
        equation = group['equation'][()].decode('utf-8')
        # make sure the random state is unique for each worker and each iteration
        # the worker_id can be the same inside the same batch
        random_state = f"{self.worker_id}_{torch.randint(100, (1,), generator=self.worker_rng).item()}"
        equation = f"{equation}_{random_state}"
        param = torch.tensor(group['param'][:], dtype=torch.float32)
        cond_k = torch.tensor(group['cond_k'][:], dtype=torch.float32)
        cond_v = torch.tensor(group['cond_v'][:], dtype=torch.float32)
        qoi_k = torch.tensor(group['qoi_k'][:], dtype=torch.float32)
        qoi_v = torch.tensor(group['qoi_v'][:], dtype=torch.float32)
        
        if self.cfg['kfeat'] == "sin":
            cond_k = torch.concat([torch.sin(2 * np.pi * cond_k), torch.cos(2 * np.pi * cond_k)], dim=-1)
            qoi_k = torch.concat([torch.sin(2 * np.pi * qoi_k), torch.cos(2 * np.pi * qoi_k)], dim=-1)
        elif self.cfg['kfeat'] == "x":
            pass
        else:
            raise ValueError(f"Unknown kfeat: {self.cfg.xfeature}")
        # random select from 100 pairs
        num_pairs = cond_k.shape[0]
        random_indices = torch.randperm(num_pairs, generator=self.worker_rng)
        demo_indices = random_indices[:self.cfg['demo_num']]
        quest_indices = random_indices[self.cfg['demo_num']:self.cfg['demo_num']+1]

        demo_cond_k = cond_k[demo_indices, :, :]
        demo_cond_v = cond_v[demo_indices, :, :]
        demo_qoi_k = qoi_k[demo_indices, :, :]
        demo_qoi_v = qoi_v[demo_indices, :, :]
        quest_cond_k = cond_k[quest_indices, :, :]
        quest_cond_v = cond_v[quest_indices, :, :]
        quest_qoi_k = qoi_k[quest_indices, :, :]
        quest_qoi_v = qoi_v[quest_indices, :, :]

        demo_cond_mask = torch.ones_like(demo_cond_k, dtype = torch.bool)[..., 0]
        demo_qoi_mask = torch.ones_like(demo_qoi_k, dtype = torch.bool)[..., 0]
        quest_cond_mask = torch.ones_like(quest_cond_k, dtype = torch.bool)[..., 0]
        quest_qoi_mask = torch.ones_like(quest_qoi_k, dtype = torch.bool)[..., 0]

        return equation, param, demo_cond_k, demo_cond_v, demo_cond_mask, \
                demo_qoi_k, demo_qoi_v, demo_qoi_mask, \
                quest_cond_k, quest_cond_v, quest_cond_mask, \
                quest_qoi_k, quest_qoi_v, quest_qoi_mask



def worker_init_fn(worker_id):
    # Ensure different seed for each worker
    seed = torch.initial_seed() % (2**32) + worker_id
    worker_rng = torch.Generator()
    worker_rng.manual_seed(seed)
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.set_worker_rng(worker_id, worker_rng)


def build_data_from_batch(batch):
    # Unpack the list of tuples from the batch
    equation, param, demo_cond_k, demo_cond_v, demo_cond_mask, \
    demo_qoi_k, demo_qoi_v, demo_qoi_mask, \
    quest_cond_k, quest_cond_v, quest_cond_mask, \
    quest_qoi_k, quest_qoi_v, quest_qoi_mask = zip(*batch)

    equation = tuple(equation)
    
    # Stack all tensor components across the batch
    param = torch.stack(param, dim=0)
    demo_cond_k = torch.stack(demo_cond_k, dim=0)
    demo_cond_v = torch.stack(demo_cond_v, dim=0)
    demo_cond_mask = torch.stack(demo_cond_mask, dim=0)
    demo_qoi_k = torch.stack(demo_qoi_k, dim=0)
    demo_qoi_v = torch.stack(demo_qoi_v, dim=0)
    demo_qoi_mask = torch.stack(demo_qoi_mask, dim=0)
    quest_cond_k = torch.stack(quest_cond_k, dim=0)
    quest_cond_v = torch.stack(quest_cond_v, dim=0)
    quest_cond_mask = torch.stack(quest_cond_mask, dim=0)
    quest_qoi_k = torch.stack(quest_qoi_k, dim=0)
    quest_qoi_v = torch.stack(quest_qoi_v, dim=0)
    quest_qoi_mask = torch.stack(quest_qoi_mask, dim=0)

    # Create the Data object
    data = Data(
        equation=equation,
        param=param,
        demo_cond_k=demo_cond_k,
        demo_cond_v=demo_cond_v,
        demo_cond_mask=demo_cond_mask,
        demo_qoi_k=demo_qoi_k,
        demo_qoi_v=demo_qoi_v,
        demo_qoi_mask=demo_qoi_mask,
        quest_cond_k=quest_cond_k,
        quest_cond_v=quest_cond_v,
        quest_cond_mask=quest_cond_mask,
        quest_qoi_k=quest_qoi_k,
        quest_qoi_mask=quest_qoi_mask,
    )
    
    label = quest_qoi_v
    
    return data, label


class DataLooper:
    def __init__(self, dataset, batch_size, num_workers, infinite):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.infinite = infinite

        self.data_loader = self.get_dataloader()
        self.data_iter = iter(self.data_loader)
        self.data_iter_num = 0

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Ensure reshuffling occurs when creating each new dataloader
            drop_last=True,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=build_data_from_batch
        )
    
    def __next__(self):
        try:
            out = next(self.data_iter)
        except StopIteration:
            if not self.infinite:
                raise StopIteration # Stop the iteration
            print(f"{self.dataset.name} reached end of data loader, restart {self.data_iter_num}")
            self.data_loader = self.get_dataloader()
            self.data_iter = iter(self.data_loader)
            self.data_iter_num += 1
            out = next(self.data_iter)
        return out

# Usage example
if __name__ == "__main__":
    data_files = "./scratch/data/data_weno_cubic/train*forward*.h5"  # HDF5 files 
    batch_size = 16
    num_workers = 4
    cfg = {"demo_num": 5, "kfeat": "sin"}

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    file_paths = glob.glob(data_files) # Get a list of all files matching the pattern
    dataset = WenoDataset(file_paths, cfg = cfg, name = "WenoDataset")

    # one epoch
    # dataloader = DataLooper(dataset, batch_size, num_workers, infinite=False)
    # while True:
    #     try:
    #         data, label = next(dataloader)
    #     except StopIteration:
    #         break

    dataloader = DataLooper(dataset, batch_size, num_workers, infinite=True)
    for i in range(10000):
        print(i, end = ",", flush=True)
        data, label = next(dataloader)
        pprint(data.equation)
        data.print_shape()
        print("label shape:", label.shape)
        print(data.get_shape_namedtuple())
        if i >= 1:
            break
