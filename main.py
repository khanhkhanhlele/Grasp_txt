# from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig

# # Initializing a CLIP-vision config
# vision_config = CLIPVisionConfig()

# # Initializing a Llama config
# text_config = LlamaConfig()

# # Initializing a Llava llava-1.5-7b style configuration
# configuration = LlavaConfig(vision_config, text_config)

# # Initializing a model from the llava-1.5-7b style configuration
# model = LlavaForConditionalGeneration(configuration)

# # Accessing the model configuration
# configuration = model.config


from trainer import trainer

import argparse
import numpy as np
import torch
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Grasp Detection', description='Grasp Detection')

    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--wk', type=str, default=1, help='number of workers')
    parser.add_argument('--pm', action='store_true', help='pin memory')
    parser.add_argument('--sz', type=int, default=224, help='size of processed image (h, w)')
    parser.add_argument('--aug', action='store_true', help='augmentation')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--idx', type=int, default=0, help='device index')
    parser.add_argument('--epoch', type=int, default=1, help='#epoch')

    # Network
    parser.add_argument('--network', type=str, default='grconvnet3',
                        help='Network name in inference/models')
    parser.add_argument("--vision-tower-name", type=str, default="openai/clip-vit-large-patch14-336",
                        help='Vision tower name in huggingface')
    parser.add_argument('--llava-model-path', type=str, default="checkpoints/llava-lora-1.11",)
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=0,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='Threshold for IOU matching')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    
    # Training
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--test', action='store_true', help='test only, skipping training')
    
    # gpt2
    parser.add_argument('--bls', type=int, default=16, help='blocksize')
    parser.add_argument('--nl', type=int, default=4, help='#layers')
    parser.add_argument('--nh', type=int, default=8, help='#heads')
    parser.add_argument('--ne', type=int, default=512, help='embedding size')

    # imgencoder
    parser.add_argument('--w', action='store_true', help='using ImageNet pretrained weight')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer(args=args)