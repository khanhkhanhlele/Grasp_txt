from torch.utils.data import Dataset, DataLoader
from .utils import imgaug, read_pickle
from glob import glob
from torch.nn.utils.rnn import pad_sequence

import os
import albumentations as A
import cv2
import torch
import numpy as np
from .preprocessing import GraspRectangles, Grasp
from LLaVA.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
)
import requests
from PIL import Image
from io import BytesIO
import json


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    image = load_image(image_files)
    out.append(image)
    return out

CURR = "/".join(__file__.split("/")[:-2])
ROOT = CURR + "/grasp-anything++"

class GAT(Dataset):
    def __init__(self, root = ROOT, train = True, img_size = 224, tokenizer=None, processor=None, model_config=None, aug=False) -> None:
        super().__init__()

        if train:
            self.part = 'seen'
        else:
            self.part = 'unseen'

        self.aug = A.Compose(imgaug(), p = 0.9, bbox_params=A.BboxParams(format='yolo'))
        self.res = A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        
        self.dir = root + f"/{self.part}"

        self.inss = glob(self.dir + "/grasp_instructions/*")
        self.lbls = [self.dir + f"/grasp_label/{os.path.basename(x).replace('.pkl', '')}.pt" for x in self.inss]
        self.imgs = [self.dir + f"/image/{os.path.basename(x).split('_')[0]}.jpg" for x in self.inss]
        
        self.aug = aug
        self.tr = train
        self.sz = img_size
        self.processor = processor
        self.model_config = model_config
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.inss)

    def __getitem__(self, index):
        ins_path = self.inss[index]
        lbl_path = self.lbls[index]
        img_path = self.imgs[index]

        img = cv2.imread(img_path)
        W, H, _ = img.shape
        x, y, w, h, a = self.lblproc(lbl_path)
        ins = read_pickle(ins_path)

        if self.tr and self.aug:
            aug_transformed = self.aug(image=img)
            transformed = self.res(image = aug_transformed['image'])
            transformed_image = transformed['image']
        else:
            transformed = self.res(image=img)
            transformed_image = transformed['image']

        # tokens = torch.tensor([self.vocab[token] for token in self.tokenizer(ins)], dtype=torch.long)
        with open("dataset/train/dataset.json", "r") as f:
            dataset = json.load(f)
        
        for data in dataset:
            if img_path.split("/")[-1].split(".")[0] in data["id"]:
                response = data["conversations"][1]["value"]
            else:
                response = "here is the object [SPT] object name [SPT]"
        
        input_ids = (
            tokenizer_image_token(ins + response, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        ) 
        
        pad_seq = torch.full((1, 511-input_ids.shape[1]), 2)
        pad_final = torch.full((1, 1), 2)
        padded_input_ids = torch.cat([pad_seq.to(input_ids.device), input_ids, pad_final.to(input_ids.device)], dim=1).squeeze(0)
        attn_mask = torch.zeros_like(pad_seq[0])
        attn_mask = torch.cat([attn_mask, torch.ones_like(input_ids[0]).cpu(), torch.zeros((1,))], dim=0)
        attn_mask = attn_mask.to(input_ids.device)
        
        images = load_images(img_path)
        images_tensor = process_images(
            images,
            self.processor,
            self.model_config
        )
        image_tensor = images_tensor[0]
        
        img = torch.from_numpy(transformed_image).permute(-1, 0, 1).float()
        lbl = torch.from_numpy(np.array([x/W, y/W, w/W, h/W, a])).float()
        x,y,w,h,theta = lbl.tolist()
        
        grs = [Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr]
        gr = GraspRectangles(grs)
        pos_img, ang_img, width_img = gr.draw((H, W))
        cos = torch.tensor(np.cos(2 * ang_img))
        sin = torch.tensor(np.sin(2* ang_img))
        pos_img = torch.tensor(pos_img)
        width_img = torch.tensor(width_img)
        lbl = torch.stack([pos_img, cos, sin, width_img], dim=0).float()
        
        return img, attn_mask, padded_input_ids, lbl, image_tensor, index

    def get_gtbb(self, didx, rot=0, zoom=1):
        grs_out = []
        for idx in didx.cpu().numpy().tolist():
            lbl_path = self.lbls[idx]
            img_path = self.imgs[idx]
            img = cv2.imread(img_path)
            W, H, _ = img.shape
            x, y, w, h, a = self.lblproc(lbl_path)
            x = x / W
            y = y / W
            w = w / W
            h = h / W
            grs_out.append(Grasp(np.array([y, x]), -a / 180.0 * np.pi, w, h).as_gr)
            
        grs = GraspRectangles(grs_out)
        # grs.rotate(rot, (H // 2, W // 2))
        # grs.zoom(zoom, (H // 2, W // 2))
        return grs
    

    @staticmethod
    def lblproc(path):
        data = torch.load(path)
        quality = data[:, 0]
        ext_data = data[quality.argmax()].tolist()
        return ext_data[1:]

def get_data(args, llava_tokenizer, llava_image_processor, model_config):

    train_ds = GAT(train=True, img_size=args.sz, aug=args.aug, tokenizer = llava_tokenizer, processor = llava_image_processor, model_config=model_config)
    valid_ds = GAT(train=True, img_size=args.sz, aug=args.aug, tokenizer = llava_tokenizer, processor = llava_image_processor, model_config=model_config)
    
    def generate_batch(data_batch):
        lbl_batch = []
        input_ids_batch = []
        image_tensor_batch = []
        attn_masks = []
        imgs = []
        idxes = []
        for (img, attn_mask, input_ids, lbl, image_tensors, idx) in data_batch:
            attn_masks.append(attn_mask)
            lbl_batch.append(lbl)
            input_ids_batch.append(input_ids)
            image_tensor_batch.append(image_tensors)
            imgs.append(img)
            idxes.append(idx)
        
        attn_masks = torch.stack(attn_masks)
        lbl_batch = torch.stack(lbl_batch)
        input_ids_batch = torch.stack(input_ids_batch)
        image_tensor_batch = torch.stack(image_tensor_batch)
        images = torch.stack(imgs)
        idxes = torch.tensor(idxes)
        return images, attn_masks, input_ids_batch, lbl_batch, image_tensor_batch, idxes

    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=generate_batch, pin_memory=args.pm)
    valid_ld = DataLoader(valid_ds, batch_size=1, shuffle=True, collate_fn=generate_batch, pin_memory=args.pm)

    return args, train_ld, valid_ld