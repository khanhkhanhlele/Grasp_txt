from torch.utils.data import Dataset, DataLoader
from .utils import imgaug, read_pickle
from glob import glob
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer

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


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

CURR = "/".join(__file__.split("/")[:-1])
ROOT = CURR + "/src"

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

        self.vocab_path = CURR + "/vocab.pkl"
        if not os.path.exists(self.vocab_path):
            raise ValueError(f"No Vocab Found at {self.vocab_path}")

        self.aug = aug
        self.tr = train
        self.sz = img_size
        self.processor = processor

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
        
        input_ids = (
            tokenizer_image_token(ins, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        
        images = load_images(img_path)
        images_tensor = process_images(
            images,
            self.processor,
            self.model_config
        )
        image_tensor = images_tensor[0]
        
        img = torch.from_numpy(transformed_image).permute(-1, 0, 1).float()
        lbl = torch.from_numpy(np.array([x/W, y/W, w/W, h/W, a/180])).float()
        x,y,w,h,theta = lbl.tolist()
        
        grs = [Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr]
        gr = GraspRectangles(grs)
        pos_img, ang_img, width_img = gr.draw((H, W))
        cos = torch.tensor(np.cos(2 * ang_img))
        sin = torch.tensor(np.sin(2* ang_img))
        pos_img = torch.tensor(pos_img)
        width_img = torch.tensor(width_img)
        lbl = torch.stack([pos_img, cos, sin, width_img], dim=0).float()
        
        return img, input_ids, lbl, image_tensor, image_tensor

    @staticmethod
    def lblproc(path):
        data = torch.load(path)
        quality = data[:, 0]
        ext_data = data[quality.argmax()].tolist()
        return ext_data[1:]

def get_data(args, llava_tokenizer, llava_image_processor):

    train_ds = GAT(train=True, img_size=args.sz, aug=args.aug, tokenizer = llava_tokenizer, processor = llava_image_processor)
    valid_ds = GAT(train=False, img_size=args.sz, aug=args.aug, tokenizer = llava_tokenizer, processor = llava_image_processor)

    def generate_batch(data_batch):
        lbl_batch = []
        img_batch = []
        input_ids_batch = []
        image_tensor_batch = []
        for (img, lbl, input_ids, image_tensors) in data_batch:
            img_batch.append(img)
            lbl_batch.append(lbl)
            input_ids_batch.append(input_ids)
            image_tensor_batch.append(image_tensors)
            
        img_batch = torch.stack(img_batch)
        lbl_batch = torch.stack(lbl_batch)
        input_ids_batch = torch.stack(input_ids_batch)
        image_tensor_batch = torch.stack(image_tensor_batch)
        return img_batch, input_ids_batch, lbl_batch, image_tensor_batch

    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=generate_batch, pin_memory=args.pm)
    valid_ld = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, collate_fn=generate_batch, pin_memory=args.pm)

    return args, train_ld, valid_ld