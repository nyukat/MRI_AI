import os
import time
import argparse
import traceback
import pickle
import random
import logging
import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, MultiStepLR
from apex import amp

from utilities.callbacks import History, RedundantCallback, resolve_callbacks, EvaluateEpoch
from utilities.warmup import GradualWarmupScheduler
from utilities.utils import boolean_string, save_checkpoint
from sampler import BPESampler, PositiveSampler
from dataset import MRIDataset
from networks import losses
from networks.models import MRIModels, MultiSequenceChannels



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def initialize_wandb(parameters):
    run = wandb.init(
        project="PROJECT-NAME",
        entity="ENTITY-NAME",
        dir=parameters.logdir,
        name=parameters.experiment,
        config=parameters
    )
    
    return run


def get_scheduler(parameters, optimizer, train_loader):
    if parameters.scheduler == 'plateau':
        scheduler_main = ReduceLROnPlateau(
            optimizer,
            patience=parameters.scheduler_patience,
            verbose=True,
            factor=parameters.scheduler_factor,
            min_lr=0
        )
    elif parameters.scheduler == 'cosineannealing':
        cosine_max_epochs = parameters.cosannealing_epochs
        if parameters.minimum_lr is not None:
            cosine_min_lr = parameters.minimum_lr
        else:
            if parameters.lr <= 1e-5:
                cosine_min_lr = parameters.lr * 0.1
            else:
                cosine_min_lr = 1e-6
        scheduler_main = CosineAnnealingLR(
            optimizer,
            T_max=(cosine_max_epochs * len(train_loader)),
            eta_min=cosine_min_lr
        )
    elif parameters.scheduler == 'step':
        scheduler_main = StepLR(optimizer, step_size=parameters.step_size, gamma=0.1)
    elif parameters.scheduler == 'stepinf':
        scheduler_main = StepLR(optimizer, step_size=999, gamma=0.1)
    elif parameters.scheduler == 'singlestep':
        scheduler_main = MultiStepLR(
            optimizer,
            milestones=[np.random.randint(30,36)],
            gamma=0.1
        )
    elif parameters.scheduler == 'multistep':
        step_size_z = random.randint((-(parameters.step_size // 2)), (parameters.step_size // 2))
        scheduler_main = MultiStepLR(
            optimizer,
            milestones=[
                parameters.step_size,
                2*parameters.step_size+step_size_z,
                3*parameters.step_size+step_size_z,
            ],
            gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler {parameters.scheduler}")
    if not parameters.warmup:
        scheduler = scheduler_main
    else:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=parameters.stop_warmup_at_epoch,
            after_scheduler=scheduler_main
        )
    
    return scheduler, scheduler_main


def train(parameters: dict, callbacks: list = None):
    # Devices & DDP
    if parameters.ddp:
        logger.info("Setting up DDP for local rank ", parameters.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = f'cuda:{parameters.local_rank}'
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(parameters.seed)
    else:
        device = torch.device("cuda")
    
    # Reproducibility & benchmarking
    torch.backends.cudnn.benchmark = parameters.cudnn_benchmarking
    torch.backends.cudnn.deterministic = parameters.cudnn_deterministic
    torch.manual_seed(parameters.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)

    # Neptune/Wandb logger
    if parameters.use_neptune and (not parameters.ddp or (parameters.ddp and parameters.local_rank == 0)):
        neptune_experiment = initialize_wandb(parameters)
    else:
        neptune_experiment = None
    
    # Load data for subgroup statistics
    subgroup_df = pd.read_pickle(parameters.subgroup_data)
    parameters.subgroup_df = subgroup_df
    
    # Prepare datasets
    train_dataset = MRIDataset(parameters, "training")
    validation_dataset = MRIDataset(parameters, "validation")

    # Sampler
    if parameters.sampler == 'none':
        sampler = None
    elif parameters.sampler =='match_bpe':
        sampler = BPESampler(train_dataset.data_list)
    elif parameters.sampler == 'positive':
        sampler = PositiveSampler(train_dataset.data_list)
    else:
        raise ValueError(f"Unknown sampler requested: {parameters.sampler}")

    # DataLoaders
    if parameters.ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        validation_sampler = DistributedSampler(validation_dataset, shuffle=False)
        train_shuffle = False
    else:
        train_sampler = sampler
        validation_sampler = None
        if parameters.sampler == 'none':
            train_shuffle = True
        else:
            train_shuffle = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=parameters.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=parameters.num_workers,
        pin_memory=parameters.pin_memory,
        drop_last=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=parameters.num_workers,
        pin_memory=parameters.pin_memory,
        drop_last=True
    )
    validation_labels = validation_dataset.get_labels()

    if parameters.lms:
        torch.cuda.set_enabled_lms(parameters.lms)  # large model support

    if parameters.architecture == 'multi_channel':
        if parameters.age_as_channel:
            in_channels = 4
        else:
            in_channels = 3
        model = MultiSequenceChannels(
            parameters,
            in_channels=in_channels,
            inplanes=parameters.inplanes,
            wide_factor=parameters.resnet_width,
            stochastic_depth_rate=parameters.stochastic_depth_rate,
            use_se_layer=parameters.use_se_layer
        )
        if parameters.weights:
            if parameters.weights_policy == 'kinetics':
                weights = torch.load(parameters.weights)
                new_weights = {}
                for k, v in weights.items():
                    if not any(exclusion in k for exclusion in ['num_batches_tracked']):
                        k_ = k.replace("layer", "resnet.layer")
                        k_ = k_.replace("conv1.0", "conv1")
                        k_ = k_.replace("conv2.0", "conv2")
                        new_weights[k_] = weights[k]
                model.load_state_dict(new_weights, strict=False)
                logger.info("Using Kinetics weights")
            else:
                raise NotImplementedError()
    else:
        model = MRIModels(parameters, inplanes=parameters.inplanes).model

    # Loss function and optimizer
    if parameters.architecture in ['3d_resnet18_fc', '2d_resnet50']:
        if parameters.label_type == 'cancer':
            loss_train = loss_eval = nn.BCEWithLogitsLoss()
        else:
            # for BI-RADS and BPE pretraining use softmax
            loss_train = loss_eval = nn.CrossEntropyLoss()
    else:
        loss_train = losses.BCELossWithSmoothing(smoothing=parameters.label_smoothing)
        loss_eval = losses.bce
    
    if parameters.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=parameters.lr,
            weight_decay=parameters.weight_decay
        )
    elif parameters.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), parameters.lr)
    elif parameters.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), parameters.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer {parameters.optimizer}")

    # Scheduler
    scheduler, scheduler_main = get_scheduler(parameters, optimizer, train_loader)

    model.to(device)
    if parameters.half:
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=parameters.half_level,
            min_loss_scale=128
        )
    
    if parameters.weights_policy == 'resume':
        # Load optimizer state if resuming
        optimizer.load_state_dict(weights['optimizer'])
        #amp.load_state_dict(weights['amp'])
    
    if parameters.ddp:
        model = DDP(
            model,
            device_ids=[parameters.local_rank],
            output_device=parameters.local_rank,
            find_unused_parameters=True,  # ?
        )
    
    # Training/validation loop
    global_step = 0
    global_step_val = 0
    best_epoch = 0
    best_metric = 0
    resolve_callbacks('on_train_start', callbacks)
    try:
        for epoch_number in tqdm(range(1, (parameters.num_epochs + 1))):
            resolve_callbacks('on_epoch_start', callbacks, epoch_number)
            last_epoch = global_step // len(train_loader)

            logger.info(f'Starting *training* epoch number {epoch_number}')
            
            if parameters.use_neptune and (not parameters.ddp or (parameters.ddp and parameters.local_rank == 0)):
                wandb.log({"epoch_number": epoch_number})

            if parameters.ddp and parameters.local_rank != 0:
                # On slave processes don't store all details
                epoch_data = {
                    "epoch_number": epoch_number
                }
            else:
                epoch_data = {
                    "epoch_number": epoch_number,
                    "subgroup_df": subgroup_df
                }
            
            training_labels = {}

            # Training phase
            if parameters.skip_training is False:
                epoch_loss = []
                model.train()

                if parameters.ddp:
                    dist.barrier()  # sync up processes before new epoch
                    torch.cuda.empty_cache()
                resolve_callbacks('on_train_start', callbacks)

                minibatch_number = 0
                number_of_used_training_examples = 0

                training_losses = []
                training_predictions = dict()

                for i_batch, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                    resolve_callbacks('on_batch_start', callbacks)
                    try:
                        indices, raw_data_batch, label_batch, mixed_label = batch
                        
                        for ind_n, ind in enumerate(indices):
                            training_labels[ind] = list(label_batch[ind_n].numpy())
                        label_batch = label_batch.to(device)

                        minibatch_loss = 0

                        if(len(label_batch) > 0):
                            number_of_used_training_examples = number_of_used_training_examples + len(label_batch)
                            subtraction = raw_data_batch  # (b_s, z, x, y)
                            
                            for param in model.parameters():
                                param.grad = None
                            if parameters.mixup:
                                mixed_label = mixed_label.to(device)
                                mixup_loss1 = loss_train(output, mixed_label[:,0,:])
                                mixup_loss2 = loss_train(output, mixed_label[:,1,:])
                                minibatch_loss = (0.5 * mixup_loss1) + (0.5 * mixup_loss2)
                            else:
                                if parameters.architecture in ['3d_resnet18_fc', '2d_resnet50']:
                                    if parameters.label_type == 'cancer':
                                        minibatch_loss = loss_train(output, label_batch.type_as(output))
                                    else:
                                        minibatch_loss = loss_train(output, torch.max(label_batch, 1)[1])  # target converted from one-hot to (batch_size, C)
                                elif parameters.architecture == '3d_gmic':
                                    is_malignant = label_batch[0][1] or label_batch[0][3]
                                    is_benign = label_batch[0][0] or label_batch[0][2]
                                    target = torch.tensor([[is_malignant, is_benign]]).cuda()
                                    minibatch_loss = loss_train(output, target)
                                else:
                                    # THIS IS THE DEFAULT LOSS
                                    minibatch_loss = loss_train(output, label_batch)
                                    #print("Loss:", minibatch_loss)
                                logger.info(f"Minibatch loss: {minibatch_loss}")
                            epoch_loss.append(float(minibatch_loss))

                            if parameters.use_neptune:
                                if parameters.ddp:
                                    pass
                                else:
                                    if global_step % parameters.log_every_n_steps == 0:
                                        wandb.log({"train/nll": minibatch_loss, "global_step": global_step})
                            
                            # Epoch-level average loss
                            if not parameters.ddp:
                                training_losses.append(minibatch_loss.item())
                            
                            # Backprop
                            if parameters.half:
                                with amp.scale_loss(minibatch_loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                minibatch_loss.backward()

                            # Optimizer
                            optimizer.step()

                            for i in range(0, len(label_batch)):
                                training_predictions[indices[i]] = output[i].cpu().detach().numpy()

                            minibatch_number += 1
                            global_step += 1
                            
                            # Log learning rate
                            current_lr = optimizer.param_groups[0]['lr']
                            if not parameters.ddp or (parameters.ddp and parameters.local_rank == 0):
                                if parameters.use_neptune:
                                    if global_step % parameters.log_every_n_steps == 0:
                                        wandb.log({"learning_rate": current_lr, "global_step": global_step})

                            # Resolve schedulers at step
                            if type(scheduler) == CosineAnnealingLR:
                                scheduler.step()
                            
                            # Warmup scheduler step update
                            if type(scheduler) == GradualWarmupScheduler:
                                if parameters.warmup and epoch_number < parameters.stop_warmup_at_epoch:
                                    scheduler.step(epoch_number + ((global_step - last_epoch * len(train_loader)) / len(train_loader)))
                                else:
                                    if type(scheduler_main) == CosineAnnealingLR:
                                        scheduler.step()
                            
                        else:
                            logger.warn('No examples in this training minibatch were correctly loaded.')
                    except Exception as e:
                        logger.error('[Error in train loop', traceback.format_exc())
                        logger.error(e)
                        continue
                    
                    resolve_callbacks('on_batch_end', callbacks)
                
                # Resolve schedulers at epoch
                if type(scheduler) == ReduceLROnPlateau:
                    scheduler.step(np.mean(epoch_loss))
                elif type(scheduler) == GradualWarmupScheduler:
                    if type(scheduler_main) != CosineAnnealingLR:
                        # Don't step for cosine; cosine is resolved at iter
                        scheduler.step(epoch=(epoch_number+1), metrics=np.mean(epoch_loss))
                elif type(scheduler) in [StepLR, MultiStepLR]:
                    scheduler.step()

                # AUROC 
                epoch_data['training_losses'] = training_losses
                epoch_data['training_predictions'] = training_predictions
                epoch_data['training_labels'] = training_labels

                # Epoch average training loss
                if parameters.use_neptune and not parameters.ddp:
                    epoch_train_loss = sum(training_losses) / len(training_losses)
                    wandb.log({"train/nll_averaged": epoch_train_loss})
                    training_losses = []

                if parameters.ddp:
                    dist.barrier()  # make sure all writes are done before we calculate statistics after an epoch
                resolve_callbacks('on_train_end', callbacks, epoch=epoch_number, logs=epoch_data, neptune_experiment=neptune_experiment)
            torch.cuda.empty_cache()

            # Validation
            if parameters.ddp:
                dist.barrier()  # make sure all writes are done before we calculate statistics after an epoch
            
            resolve_callbacks('on_val_start', callbacks)
            model.eval()
            logger.info(f'Starting *validation* epoch number {epoch_number}')
            with torch.no_grad():
                minibatch_number = 0
                number_of_used_validation_examples = 0

                validation_losses = []
                validation_predictions = dict()
                
                for i_batch, batch in tqdm(enumerate(validation_loader), total=len(validation_loader)):
                    indices, raw_data_batch, label_batch, _ = batch
                    global_step_val += 1

                    label_batch = label_batch.to(device)

                    if len(label_batch) > 0:
                        number_of_used_validation_examples = number_of_used_validation_examples + len(label_batch)
                        
                        subtraction = raw_data_batch
                        
                        if parameters.architecture in ['r2plus1d_18', 'mc3_18'] and parameters.input_type != 'three_channels':
                            subtraction = subtraction.unsqueeze(1).contiguous()
                        
                        if parameters.input_type == 'random':
                            modality_losses = []
                            for x_modality in range(subtraction.shape[1]):
                                x = subtraction[:, x_modality, ...]
                                output = model(x.to(device))
                                modality_loss = loss_eval(output, label_batch)
                                modality_losses.append(modality_loss.item())
                            minibatch_loss = sum(modality_losses) / len(modality_losses)
                            if not parameters.ddp:
                                validation_losses.append(minibatch_loss)
                        else:
                            if parameters.architecture == '3d_gmic':
                                output, _ = model(subtraction.to(device))
                            else:
                                # default
                                output = model(subtraction.to(device))

                            if parameters.architecture in ['3d_resnet18_fc', '2d_resnet50']:
                                if parameters.label_type == 'cancer':
                                    minibatch_loss = loss_eval(output, label_batch.type_as(output))
                                else:
                                    minibatch_loss = loss_eval(output, torch.max(label_batch, 1)[1])  # target converted from one-hot to (batch_size, C)
                            elif parameters.architecture == '3d_gmic':
                                is_malignant = label_batch[0][1] or label_batch[0][3]
                                is_benign = label_batch[0][0] or label_batch[0][2]
                                target = torch.tensor([[is_malignant, is_benign]]).cuda()
                                minibatch_loss = loss_eval(output, target)
                            else:
                                # DEFAULT LOSS IN VAL
                                minibatch_loss = loss_eval(output, label_batch)                        
                            if not parameters.ddp:
                                validation_losses.append(minibatch_loss.item())
                        logger.info(f"Minibatch loss: {minibatch_loss}")
                        
                        if parameters.use_neptune:
                            if not parameters.ddp:
                                if global_step_val % parameters.log_every_n_steps == 0:
                                    wandb.log({"val/nll": minibatch_loss, "global_step_val": global_step_val})

                        for i in range(0, len(label_batch)):
                            validation_predictions[indices[i]] = output[i].cpu().numpy()

                    minibatch_number = minibatch_number + 1

                if parameters.use_neptune and not parameters.ddp:
                    epoch_val_loss = sum(validation_losses) / len(validation_losses)
                    wandb.log({"val/nll_averaged": epoch_val_loss})
                epoch_data['validation_losses'] = validation_losses
                epoch_data['validation_predictions'] = validation_predictions
                epoch_data['validation_labels'] = validation_labels
                validation_losses = []

                torch.cuda.empty_cache()
            
            if parameters.ddp:
                dist.barrier()  # make sure all writes are done before we calculate statistics after an epoch
            val_res = resolve_callbacks('on_val_end', callbacks, epoch=epoch_number, logs=epoch_data, neptune_experiment=neptune_experiment)
            
            # Checkpointing
            if (parameters.ddp and parameters.local_rank != 0) or not parameters.save_checkpoints:
                # Do not save checkpoints if distributed slave process or user specifies an arg
                pass
            else:
                if parameters.save_best_only:
                    if parameters.label_type == 'birads' or parameters.label_type == 'bpe':
                        birads_AUC = val_res['EvaluateEpoch']['auc']
                        if birads_AUC > best_metric:
                            best_metric = birads_AUC
                            best_epoch = epoch_number
                            model_file_name = os.path.join(parameters.model_saves_directory, f"model_best_auroc")
                            save_checkpoint(model, model_file_name, optimizer, is_amp=parameters.half, epoch=epoch_number)
                    else:
                        malignant_AUC = val_res['EvaluateEpoch']['auc_malignant']
                        if malignant_AUC > best_metric:
                            best_metric = malignant_AUC
                            best_epoch = epoch_number
                            model_file_name = os.path.join(parameters.model_saves_directory, f"model_best_auroc")
                            save_checkpoint(model, model_file_name, optimizer, is_amp=parameters.half, epoch=epoch_number)
                else:
                    model_file_name = os.path.join(parameters.model_saves_directory, f"model-epoch{epoch_number}")
                    save_checkpoint(model, model_file_name, optimizer, step=global_step, is_amp=parameters.half, epoch=epoch_number)
            
            if parameters.ddp:
                dist.barrier()
            resolve_callbacks('on_epoch_end', callbacks, epoch=epoch_number, logs=epoch_data)
    except KeyboardInterrupt:
        if parameters.use_neptune:
            wandb.finish()
        if parameters.ddp:
            torch.distributed.destroy_process_group()
    
    if parameters.use_neptune:
        wandb.finish()
    if parameters.ddp:
        torch.distributed.destroy_process_group()

    return


def get_args():
    parser = argparse.ArgumentParser("MRI Training pipeline")

    # File paths
    parser.add_argument("--metadata", type=str, default="/PATH/TO/PICKLE/FILE/WITH/METADATA.pkl", help="Pickled metadata file path")
    parser.add_argument("--datalist", type=str, default="/PATH/TO/PICKLE/FILE/WITH/DATALIST.pkl", help="Pickled data list file path")
    parser.add_argument("--subgroup_data", type=str, default='/PATH/TO/PICKLE/FILE/WITH/SUBGROUP/DATA.pkl', help='Pickled data with subgroup information')
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--weights_policy", type=str, help='Custom loaded weights surgery')

    # Input   
    parser.add_argument("--input_type", type=str, default='sub_t1c2', choices={'sub_t1c1', 'sub_t1c2', 't1c1', 't1c2', 't1pre', 'mip_t1c2', 'three_channel', 't2', 'random', 'multi', 'MIL'})
    parser.add_argument("--input_size", type=str, default='normal', choices={'normal', 'small'})
    parser.add_argument("--label_type", type=str, default='cancer', choices={'cancer', 'birads', 'bpe'}, help='What labels should be used, e.g. pretraining on BIRADS and second stage on cancer.')
    parser.add_argument("--sampler", type=str, default='none', choices={'normal', 'match_bpe', 'positive'})
    parser.add_argument("--subtraction_clipping", type=boolean_string, default=False, help='When performing subtraction, clip lower range values to 0')
    parser.add_argument("--preprocessing_policy", type=str, default='none', choices={'none', 'clahe'})
    parser.add_argument("--age_as_channel", type=boolean_string, default=False, help='Use age as additional channel')
    parser.add_argument("--isotropic", type=boolean_string, default=False, help='Use isotropic spacing (default is False-anisotropic)')

    # Model & augmentation
    parser.add_argument("--architecture", type=str, default="3d_resnet18", choices={'3d_resnet18', '3d_gmic', '3d_resnet18_fc', '3d_resnet34', '3d_resnet50', '3d_resnet101', 'r2plus1d_18', 'mc3_18', '2d_resnet50', 'multi_channel'})
    parser.add_argument("--resnet_width", type=float, default=1, help='Multiplier of ResNet width')
    parser.add_argument("--topk", type=int, default=10, help='Used only in our modified 3D resnet')
    parser.add_argument("--resnet_groups", type=int, default=16, help='Set 0 for batch norm; otherwise value will be number of groups in group norm')
    parser.add_argument("--aug_policy", type=str, default='affine', choices={'affine', 'none', 'strong_affine', 'rare_affine', 'weak_affine', 'policy1', '5deg_10scale', '10deg_5scale', '10deg_10scale', '10deg_10scale_p75', 'motion', 'ghosting', 'spike'})
    parser.add_argument("--affine_scale", type=float, default=0.10)
    parser.add_argument("--affine_rotation_deg", type=int, default=10)
    parser.add_argument("--affine_translation", type=int, default=0)
    parser.add_argument("--mixup", type=boolean_string, default=False, help='Use mixup augmentation')
    parser.add_argument("--loss", type=str, default="bce", choices={'bce'}, help='Which loss function to use')
    parser.add_argument("--network_modification", type=str, default=None, choices={'resnet18_bottleneck', 'resnet_36'})
    parser.add_argument("--cutout", type=boolean_string, default=False, help='Apply 3D cutout at training')
    parser.add_argument("--cutout_percentage", type=float, default=0.4)
    parser.add_argument("--label_smoothing", type=float, default=0.0, help='Label smoothing ratio')
    parser.add_argument("--dropout", type=boolean_string, default=False, 
    help='Adds Dropout layer before FC layer with p=0.25.')
    parser.add_argument("--stochastic_depth_rate", type=float, default=0.0,
    help='Uses stochastic depth in training')
    parser.add_argument("--use_se_layer", type=boolean_string, default=False,
    help='Use squeeze-and-excitation module')
    parser.add_argument("--inplanes", type=int, default=64)

    # Parallel computation
    parser.add_argument("--ddp", type=boolean_string, default=False, help='Use DistributedDataParallel')
    parser.add_argument("--gpus", type=int, default=1, help='Needs to be specified when using DDP')
    parser.add_argument("--local_rank", type=int, default=-1, metavar='N', help='Local process rank')

    # Optimizers, schedulers
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=75)
    parser.add_argument("--optimizer", type=str, default='adam', choices={'adam', 'adamw', 'sgd'})
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=boolean_string, default=True)
    parser.add_argument("--stop_warmup_at_epoch", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default='cosineannealing', choices={'plateau', 'cosineannealing', 'step', 'stepinf', 'multistep', 'singlestep'})
    parser.add_argument("--step_size", type=int, default=15, help='If using StepLR, this is a step_size value')
    parser.add_argument("--scheduler_patience", type=int, default=7, help='Patience for ReduceLROnPlateau')
    parser.add_argument("--scheduler_factor", type=float, default=0.1, help='Rescaling factor for scheduler')
    parser.add_argument("--cosannealing_epochs", type=int, default=60, help='Length of a cosine annealing schedule')
    parser.add_argument("--minimum_lr", type=float, default=None, help='Minimum learning rate for the scheduler')

    # Efficiency 
    parser.add_argument("--num_workers", type=int, default=19)
    parser.add_argument("--pin_memory", type=boolean_string, default=True)
    parser.add_argument("--cudnn_benchmarking", type=boolean_string, default=True)
    parser.add_argument("--cudnn_deterministic", type=boolean_string, default=False)
    parser.add_argument("--half", type=boolean_string, default=True, help="Use half precision (fp16)")
    parser.add_argument("--half_level", type=str, default='O2', choices={'O1', 'O2'})
    parser.add_argument("--lms", type=boolean_string, default=False, help='Use Large Model Support')
    parser.add_argument("--training_fraction", type=float, default=1.00)
    parser.add_argument("--number_of_training_samples", type=int, default=None, help='If this value is not None, it will overrule `training_fraction` parameter')
    parser.add_argument("--validation_fraction", type=float, default=1.00)
    parser.add_argument("--number_of_validation_samples", type=int, default=None, help='If this value is not None, it will overrule `validation_fraction` parameter')

    # Logging & debugging
    parser.add_argument("--logdir", type=str, default="/DIR/TO/LOGS/", help="Directory where logs are saved")
    parser.add_argument("--experiment", type=str, default="mri_training", help="Name of the experiment that will be used in logging")
    parser.add_argument("--skip_training", type=boolean_string, default=False, help="Validation only")
    parser.add_argument("--use_neptune", type=boolean_string, default=True)
    parser.add_argument("--log_every_n_steps", type=int, default=30)
    parser.add_argument("--save_checkpoints", type=boolean_string, default=True, help='Set to False if you dont want to save checkpoints')
    parser.add_argument("--save_best_only", type=boolean_string, default=False, help='Save checkpoints after every epoch if True; only after metric improvement if False')
    parser.add_argument("--seed", type=int, default=420)
    
    args = parser.parse_args()

    if args.ddp:
        if not torch.distributed.is_available():
            raise ValueError("Pytorch distributed package is not available")
        args.is_master = args.local_rank == 0
        args.device = torch.cuda.device(args.local_rank)
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training fractions
    assert (0 < args.training_fraction <= 1.00), "training_fraction not in (0,1] range."
    assert (0 < args.validation_fraction <= 1.00), "validation_fraction not in (0,1] range."

    # Logging directories
    args.experiment_dirname = args.experiment + time.strftime('%Y%m%d%H%M%S')
    args.model_saves_directory = os.path.join(args.logdir, args.experiment_dirname)
    if os.path.exists(args.model_saves_directory):
        print("Warning: This model directory already exists")
    os.makedirs(args.model_saves_directory, exist_ok=True)

    # Save all arguments to the separate file
    if not args.ddp or (args.ddp and args.local_rank == 0):
        parameters_path = os.path.join(args.model_saves_directory, "parameters.pkl")
        with open(parameters_path, "wb") as f:
            pickle.dump(vars(args), f)

    return args


def set_up_logger(args, log_args=True):
    if args.ddp:
        log_file_name = f'output_log_{args.local_rank}.log'
    else:
        log_file_name = 'output_log.log'
    log_file_path = os.path.join(args.model_saves_directory, log_file_name)
    
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logger.info(f"model_dir = {args.model_saves_directory}")
    if log_args:
        args_pprint = pprint.pformat(args)
        logger.info(f"parameters:\n{args_pprint}")

    return


if __name__ == "__main__":
    args = get_args()
    set_up_logger(args)
    callbacks = [
        History(
            save_path=args.model_saves_directory,
            distributed=args.ddp,
            local_rank=args.local_rank
        ),
        RedundantCallback(),
        EvaluateEpoch(
            save_path=args.model_saves_directory,
            distributed=args.ddp,
            local_rank=args.local_rank,
            world_size=args.gpus,
            label_type=args.label_type
        )
    ]
    train(args, callbacks)
