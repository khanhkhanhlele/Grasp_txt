from src.ds import get_data
from src.models import get_network
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
<<<<<<< HEAD

=======
>>>>>>> f52f0d360db3fbd7845691dab64f87c97cbf86fa
# from metrics import probiou

import torch
import wandb
import json
import hashlib
import os
import logging

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
        for attn_mask, input_ids, y, image_tensor in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            
            features = vision_tower(image_tensor)
            language_features = llava(input_ids, image_tensor, attn_mask)
            
            fused_features = features + language_features
            
            xc = fused_features
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 5 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
<<<<<<< HEAD
                # logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
=======
>>>>>>> f52f0d360db3fbd7845691dab64f87c97cbf86fa

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


    # old_valid_loss = 1e26
    for epoch in range(args.epoch):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_ld, optimizer, args.batches_per_epoch)
        
        # Log training losses to tensorboard
        # tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        # for n, l in train_results['losses'].items():
        #     tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.iou_threshold)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))

        # Log validation results to tensorbaord
        # tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        # tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        # for n, l in test_results['losses'].items():
        #     tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            best_iou = iou


        save_dict = {
            'model_state_dict': net.state_dict()
        }
        torch.save(save_dict, last_model_path)
