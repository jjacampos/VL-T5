import dist_utils
from pprint import pformat
import numpy as np
from packaging import version
from dialog_pre_training_data import DialPreFineTuneDataset
import logging
from tqdm import tqdm
from pathlib import Path
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import json
from param import parse_args
import torch
import wandb

from utils import get_memories_mappings
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import h5py
from dist_utils import reduce_dict


from utils import load_state_dict, LossMeter, set_global_logging_level
from trainer_base import TrainerBase

_use_native_amp = False
_use_apex = False


# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


class Trainer(TrainerBase):
    def __init__(self, args, memories_to_coco_ids, coco_features, train=True):
        super().__init__(
            args,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from dialog_pre_training_model import VLT5DialPre, VLBartDialPre

        self.wandb_initialized = False
        
        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5DialPre
        elif 'bart' in args.backbone:
            model_class = VLBartDialPre
        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        num_added_toks = 0
        if config.use_vis_order_embedding:
            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                [f'<img_extra_id_{i}>' for i in range(100-1, -1, -1) ]
            special_tokens_dict = {
                'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(
                special_tokens_dict)

            config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids(
                [f'<vis_extra_id_{i}>' for i in range(100)])

        # Add COMET special tokens
        comet_special_tokens = json.load(open(args.special_tokens_path, 'r', encoding='utf-8'))
        """
        if 't5' in self.args.tokenizer:
            self.tokenizer.add_tokens('<')
        """
        comet_added_tokens = self.tokenizer.add_special_tokens(comet_special_tokens)

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.shared.num_embeddings + num_added_toks + comet_added_tokens)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(
                self.model.model.shared.num_embeddings + num_added_toks + comet_added_tokens)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None and not args.just_text_model:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        print('Building the train loader')
        train_raw_data = json.load(open(args.train_path, 'r', encoding='utf-8'))
        train_dataset = DialPreFineTuneDataset(train_raw_data, memories_to_coco_ids, coco_features, args, self.tokenizer, args.randomization)
        train_sampler = DistributedSampler(train_dataset) if args.distributed else Sampler(train_dataset)
        self.train_loader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      collate_fn=train_dataset.collate_fn)
    
        print('Building the val loader')
        val_raw_data = json.load(open(args.valid_path, 'r', encoding='utf-8'))
        val_dataset = DialPreFineTuneDataset(val_raw_data, memories_to_coco_ids, coco_features, args, self.tokenizer, 'no_random')
        self.val_loader = DataLoader(val_dataset,
                                    batch_size=args.valid_batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    sampler=None,
                                    collate_fn=val_dataset.collate_fn,
                                    drop_last=False)
        
        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        
    def train(self):

        LOSSES_NAME = self.args.LOSSES_NAME
        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_valid = 0.
            best_epoch = 0

            if not self.wandb_initialized:
                if 't5' in self.args.backbone:
                    project_name = "VLT5_DialPre"
                elif 'bart' in self.args.backbone:
                    project_name = "VLBart_DialPre"

                wandb.init(project=project_name)
                wandb.run.name = self.args.run_name
                wandb.config.update(self.args)
                wandb.watch(self.model)

                src_dir = Path(__file__).resolve().parent
                base_path = str(src_dir.parent)
                src_dir = str(src_dir)
                wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

                self.wandb_initialized = True

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        epochs = self.args.epochs

        for epoch in range(epochs):
            
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=240)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                update = True
                if self.args.gradient_accumulation_steps > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if update:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    # self.model.zero_grad()
                    for param in self.model.parameters():
                        param.grad = None
                    global_step += 1
            
                for k, v in results.items():
                    if k in epoch_results:
                        try:
                            epoch_results[k] += v.item()
                        except:
                            epoch_results[k] += v

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                pbar.close()

            # Validation
            valid_results = self.validate(self.val_loader)
            valid_results = reduce_dict(valid_results, average=False)

            if self.verbose:
                valid_loss = valid_results['total_loss']
                valid_loss_count = valid_results['total_loss_count']

                avg_valid_loss = valid_loss / valid_loss_count
                losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                for name, loss in valid_results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(valid_results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss / loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                            wandb.log({f'Valid Loss/{name}': avg_loss}, step=epoch)

                losses_str += '\n'
                print(losses_str)

                if avg_valid_loss < best_valid or epoch == 0:
                    best_valid = avg_valid_loss
                    best_epoch = epoch
                    self.save("BEST")

                log_str = ''

                log_str += pformat(valid_results)
                log_str += "\nEpoch %d: Valid loss %0.4f" % (epoch, avg_valid_loss)
                log_str += "\nEpoch %d: Best loss %0.4f\n" % (best_epoch, best_valid)

                print(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")


    def validate(self, loader):

        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        self.model.eval()
        with torch.no_grad():
            losses = []
            for i, batch in enumerate(tqdm(loader, ncols=240, desc="Prediction", disable=not self.verbose)):
                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()
                            
            
            if self.args.distributed:
                dist.barrier()

        return epoch_results
        
        
def main(gpu, args):

    args.gpu = gpu
    args.rank = gpu
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')

    # Set the coco API and the mapping from memory ids to coco ids
    print('Getting mapping memories to coco ids')
    memories_to_coco_ids = get_memories_mappings(args)

    # Load the coco image features
    coco_features = h5py.File(args.coco_features_path, 'r')

    if args.do_train:
        
        trainer = Trainer(args, memories_to_coco_ids, coco_features, train=True)
        trainer.train()
        

if __name__ == '__main__':

    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.dialog_losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss') # total loss

    args.LOSSES_NAME = LOSSES_NAME

    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        '''
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name
        '''
    if args.distributed:
        main(args.local_rank, args)
