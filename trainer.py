from src.ds import get_data
from src.models import get_network
from torch.optim import Adam
# from metrics import probiou
from transformers import CLIPVisionModel

from src import evaluation
from src.utils import post_process_output
import torch
import json
import hashlib
import os
import logging
from src.model import LLaVa


def validate(net, llava, device, val_data, iou_threshold):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)
    counter = 0 
    loss_avg = 0
    with torch.no_grad():
        for xc, attn_mask, input_ids, y, image_tensor, didx in val_data:
            if counter > 100:
                break
            language_features = llava(input_ids, image_tensor.half(), attn_mask)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc, language_features)

            loss = lossd['loss']
            loss_avg += loss.item()

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.get_gtbb(didx, 0.0, 1.0),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               threshold=iou_threshold
                                               )

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1
                
            counter += 1
        print('Validation Loss: {:0.4f}'.format(loss_avg / counter))
    return results

def train(epoch, net, vision_tower, llava, device, train_data, optimizer, batches_per_epoch):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        for xc, attn_mask, input_ids, y, image_tensor, idx in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            
            # features = vision_tower(image_tensor.to(device))["last_hidden_state"]
            language_features = llava(input_ids, image_tensor.half(), attn_mask)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc, language_features)

            loss = lossd['loss']

            if batch_idx % 5 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()
                
            if batch_idx % 200 == 0:
                save_dict = {
                    'model_state_dict': net.state_dict(),
                    'llava_state_dict': llava.state_dict()
                }
                torch.save(save_dict, f"checkpoints/llava-lora-{epoch}.{batch_idx}.pt")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results

def get_hash(args):
    args_str = json.dumps(vars(args), sort_keys=True)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    return args_hash

def trainer(args):

    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")

    run_name = get_hash(args)
    
    run_dir = os.getcwd() + '/runs'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    sv_dir = run_dir + f"/{run_name}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)
    
    last_model_path = sv_dir + f'/last.pt'

    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    net = network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size
    )

    net = net.to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total Params: {total_params}")
    total_train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_train_params}")
    
    vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower_name)
    vision_tower = vision_tower.to(device)
    llava = LLaVa(args.llava_model_path)
    
    args, train_ld, valid_ld = get_data(args, llava.tokenizer, llava.image_processor, llava.model_config)

    for param in vision_tower.parameters():
        param.requires_grad = False
    
    for n, p in llava.named_parameters():
        if "lora" in n:
            p.requires_grad = True
        elif "adapter" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
            
    if os.path.exists(args.pretrained_path):
        print("Loading Pretrained Model")
        pretrain_dict = torch.load(args.pretrained_path)
        net.load_state_dict(pretrain_dict['model_state_dict'])
        llava.load_state_dict(pretrain_dict['llava_state_dict'])

    params = list(net.parameters()) + list(vision_tower.parameters()) + list(llava.parameters())
    
    optimizer = Adam(params, lr=args.lr)

    if not args.test:
        for epoch in range(args.epoch):
            print('Beginning Epoch {:02d}'.format(epoch))
            train_results = train(epoch, net, vision_tower, llava, device, train_ld, optimizer, args.batches_per_epoch)
            print('Train Results: {}'.format(train_results))

            save_dict = {
                'model_state_dict': net.state_dict(),
                'llava_state_dict': llava.state_dict()
            }
            torch.save(save_dict, last_model_path)

            # Run Validation
            print('Validating...')
            test_results = validate(net, llava, device, valid_ld, args.iou_threshold)
            print('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                        test_results['correct'] / (test_results['correct'] + test_results['failed'])))
    else:
        print('Testing...')
        test_results = validate(net, llava, device, valid_ld, args.iou_threshold)
        print('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                    test_results['correct'] / (test_results['correct'] + test_results['failed'])))
