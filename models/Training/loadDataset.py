import os
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn
import numpy as np

class flowfusionDataset(Dataset):
    def __init__(self,img_dir, transform=None, target_transform=None):
        self.samples = []
        self.classes = sorted(os.listdir(img_dir))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

        for category in self.classes:
            category_path = os.path.join(img_dir, category)

            for file_name in os.listdir(category_path):
                if file_name.endswith(".npy"):
                    self.samples.append((os.path.join(category_path, file_name), self.class_to_idx[category]))

    # Called by DataLoader
    def __len__(self):
        return len(self.samples)

    # Called by DataLoader
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path)

        tensor_data =  torch.from_numpy(data).float()
        tensor_label = torch.tensor(label)

        return tensor_data, tensor_label

def dynamic_collate_fn(batch):
    # batch is a list of tuples: [(tensor_sample_1, label_1), (tensor_sample_2, label_2), ...]
    sequences, labels = zip(*batch)
    
    # pad_sequence pads along the first dimension (T) to match the longest sequence in this batch
    # batch_first=True outputs shape: (Batch, Max_T_in_batch, C*W, V, M)
    padded_sequences = rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    
    return padded_sequences, labels