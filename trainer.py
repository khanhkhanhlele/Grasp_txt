from ds import get_data
from inference.models import get_network
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from alive_progress import alive_it
# from metrics import probiou

import torch
import wandb
import json
import hashlib
import os
import logging

def train(epoch, net, device, train_data, optimizer, batches_per_epoch):
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
        for x, _, y in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 5 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

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

    args, train_ld, valid_ld = get_data(args)

    print(f"#TRAIN Batch: {len(train_ld)}")
    print(f"#VALID Batch: {len(valid_ld)}")

    run_name = get_hash(args)
    
    run_dir = os.getcwd() + '/runs'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    sv_dir = run_dir + f"/{run_name}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)
    
    best_model_path = sv_dir + f'/best.pt'
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

    optimizer = Adam(net.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(train_ld) * args.epoch)


    # old_valid_loss = 1e26
    for epoch in range(args.epoch):
        log_dict = {}
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_ld, optimizer, args.batches_per_epoch)

        # model.train()
        # total_loss = 0
        # total_iou = 0
        # for img, txt, lbl in alive_it(train_ld):
        #     img = img.to(device)
        #     txt = txt.to(device)
        #     lbl = lbl.to(device)

        #     loss, output = model(img, lbl)
            # iou = probiou(output, lbl)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
    #         scheduler.step()
            
    #         total_loss += loss.item()
    #         total_iou += iou.item()
        
    #     train_mean_loss = total_loss / len(train_ld)
    #     train_miou = total_iou / len(train_ld)
        
    #     log_dict['train/loss'] = train_mean_loss
    #     log_dict['train/miou'] = train_miou

    #     print(f"Epoch: {epoch} - Train Loss: {train_mean_loss} - Train mIoU: {train_miou}")

    #     model.eval()
    #     with torch.no_grad():
    #         total_loss = 0
    #         total_iou = 0
    #         for img, txt, lbl in alive_it(valid_ld):
    #             img = img.to(device)
    #             txt = txt.to(device)
    #             lbl = lbl.to(device)

    #             loss, output = model(img, txt, lbl)
    #             iou = probiou(output, lbl)

    #             total_loss += loss.item()
    #             total_iou += iou.item()
        
    #     valid_mean_loss = total_loss / len(valid_ld)
    #     valid_miou = total_iou / len(valid_ld)

    #     log_dict['valid/loss'] = valid_mean_loss
    #     log_dict['valid/miou'] = valid_miou

    #     print(f"Epoch: {epoch} - Valid Loss: {valid_mean_loss} - Valid mIoU: {valid_miou}")

    #     save_dict = {
    #         'args' : args,
    #         'model_state_dict': model.state_dict()
    #     }

    #     if valid_mean_loss < old_valid_loss:
    #         old_valid_loss = valid_mean_loss
            
    #         torch.save(save_dict, best_model_path)
    #     torch.save(save_dict, last_model_path)
