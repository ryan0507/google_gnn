from typing import Optional, Callable
import os
import os.path as osp
import glob
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected


class TPUGraphsSplit(Dataset):
    def __init__(self, root: str, thres: int = 1000,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 source: str = 'xla',  # 'nlp' or 'xla'
                 search: str = 'random'  # 'random' or 'default'
                 ):
        assert source in ('nlp', 'xla')
        assert search in ('random', 'default')
        self.thres = thres
        self.source = source
        self.search = search
        super().__init__(root, transform, pre_transform, pre_filter)
        self.split_dict = self._read_split_dict()

    @property
    def raw_file_names(self):
        raw_paths = []
        for split in ["train", "valid", "test"]:
            raw_paths += glob.glob(osp.join(self.raw_dir, f'npz/layout/{self.source}/{self.search}/{split}', '*.npz'))
        return raw_paths

    @property
    def processed_file_names(self):
        return [f'{self.source}_{self.search}_data_{i}.pt' for i in range(len(self.raw_file_names))]

    def _read_split_dict(self):
        split_dict_path = osp.join(self.processed_dir, f'{self.source}_{self.search}_split_dict_segment_{self.thres}.pt')
        if osp.exists(split_dict_path):
            return torch.load(split_dict_path)
        else:
            return {'train': [], 'valid': [], 'test': []}

    def _process_file(self, raw_path, split_name):
        # Implement the processing of file here.
        # You should return a Data object.
        np_file = dict(np.load(raw_path))
        if "edge_index" not in np_file:
            print('error in', raw_path)
        edge_index = torch.tensor(np_file["edge_index"].T)
        # Set if we want to add reverse edge
        # edge_index = to_undirected(edge_index)

        shape = np_file["node_config_feat"].shape
        print(f"Before preprocessing: {shape}")
        
        if split_name == 'train' or split_name == 'valid':
            print(f'{split_name} Dataset, Erase Duplicate configurations')
            preprocess_config_feats, uni_idx = np.unique(np_file["node_config_feat"], axis=0, return_index = True)
            np_file["node_config_feat"] = preprocess_config_feats
            
        
        runtime = torch.tensor(np_file["config_runtime"])
        if split_name == 'train' or split_name == 'valid':
            runtime = torch.tensor(runtime[uni_idx])
            
        op = torch.tensor(np_file["node_feat"])
        op_code = torch.tensor(np_file["node_opcode"])
        config_feats = torch.tensor(np_file["node_config_feat"])
        processed_shape = config_feats.size()
        config_feats = config_feats.view(-1, config_feats.shape[-1])
        config_idx = torch.tensor(np_file["node_config_ids"])
        num_config = torch.tensor(np_file["node_config_feat"].shape[0])
        num_config_idx = torch.tensor(np_file["node_config_feat"].shape[1])
        num_nodes = torch.tensor(np_file["node_feat"].shape[0])
        num_parts = num_nodes // self.thres + 1
        interval = num_nodes // num_parts
        partptr = torch.arange(0, num_nodes, interval+1)
        runtime_shape = runtime.size()
        
        print(f"After duplicated values new_runtime: {runtime_shape}")
        print(f"After duplicated values config: {processed_shape}")
        
        parts_cnt = 0
        
        if partptr[-1] != num_nodes:
            partptr = torch.cat([partptr, torch.tensor([num_nodes])])
        data = Data(edge_index=edge_index, op_feats=op, op_code=op_code, config_feats=config_feats, config_idx=config_idx,
                    num_config=num_config, num_config_idx=num_config_idx, y=runtime, num_nodes=num_nodes, partptr=partptr, partition_idx = parts_cnt)
        return data

    def process(self):
        split_dict = {'train': [], 'valid': [], 'test': []}
        for i, raw_path in enumerate(self.raw_file_names):
            # Process the file and save it
            data_split_type = raw_path.split('/')[-2]
            data = self._process_file(raw_path, data_split_type)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, f'{self.source}_{self.search}_data_{i}.pt'))
            split_dict[data_split_type].append(i)
            
        # Save split_dictionary
        print(split_dict)
        torch.save(split_dict,osp.join(self.processed_dir, f'{self.source}_{self.search}_split_dict_segment_{self.thres}.pt'))
        self.split_dict = split_dict    

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data

    def get_idx_split(self):
        return self.split_dict

if __name__ == '__main__':
    dataset = TPUGraphsSplit(root='datasets/TPUGraphs', source = 'xla', search = 'random')
    print(dataset[0])  # Get the first graph object
