#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, torchvision
import json
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import MultiModalBertDataset, codedbertRandomPatchesDataset
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# directory containing all raw images
IMG_PATH = "/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/PHD-CD_Research/ADARI/images/ADARI_v2/furniture/full"
PAIRS_PATH = "/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/PHD-CD_Research/ADARI/json_files/cleaned/ADARI_v2/furniture/ADARI_furniture_pairs.json"
#IMG_PATH = "/home/alex/CMU/777/ADARI/v2/full"
#PAIRS_PATH = "/home/alex/CMU/777/ADARI/ADARI_furniture_pairs.json"
# ENCODER_CNN_PATH = "resnet_9_v4.1.pt"
ENCODER_CNN_PATH = None

def open_json(path):
    f = open(path) 
    data = json.load(f) 
    f.close()
    return data 

def save_json(file_path, data):
    out_file = open(file_path, "w")
    json.dump(data, out_file)
    out_file.close()
    
dataset = MultiModalBertDataset(IMG_PATH, PAIRS_PATH, ENCODER_CNN_PATH, device=device)
# dataset = codedbertRandomPatchesDataset(IMG_PATH, PAIRS_PATH, device=device)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

df = pd.DataFrame(columns=["patches", "input_ids", "is_paired", "attention_mask", "img_name", "patch_positions", "sent_embeds"])
for j, (patches, input_ids, is_paired, attention_mask, img_name, patch_positions, sent_embeds) in enumerate(tqdm(dataloader)):
    for i in range(patches.shape[0]):
        df = df.append(
            {
                "patches": patches[i].cpu().numpy(),
                "input_ids": input_ids[i].cpu().numpy(),
                "is_paired": is_paired[i].cpu().numpy(),
                "attention_mask": attention_mask[i].cpu().numpy(),
                "img_name": img_name[i],
                "patch_positions": patch_positions[i].cpu().numpy(),
                "sents_embeds": sent_embeds[i].cpu().numpy(),
            },
            ignore_index=True
        )
    if j % 10 == 0:
        print(f"At batch # {j}")
    

df.to_pickle("./preprocessed_np_nm_sent_embeds.pkl")

