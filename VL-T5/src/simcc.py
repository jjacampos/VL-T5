import dist_utils
from pprint import pformat
import numpy as np
from packaging import version
from simcc_data import SIMMCFineTuneDataset, SIMMCEvaluator
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
import transformers

from utils import load_state_dict, LossMeter, set_global_logging_level
from trainer_base import TrainerBase

_use_native_amp = False
_use_apex = False


# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


class Trainer(TrainerBase):
    def __init__(self, args, train=True):
        super().__init__(
            args,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from simcc_model import VLT5SIMMC, VLBartSIMMC

        self.wandb_initialized = False
        
        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5SIMMC
        elif 'bart' in args.backbone:
            model_class = VLBartSIMMC
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

        # Add SIMMC special tokens
        simmc_special_tokens = json.load(open(args.special_tokens_path, 'r', encoding='utf-8'))
        """
        if 't5' in self.args.tokenizer:
            self.tokenizer.add_tokens('<')
        """
        simmc_added_tokens = self.tokenizer.add_special_tokens(simmc_special_tokens)

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.shared.num_embeddings + num_added_toks + simmc_added_tokens)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(
                self.model.model.shared.num_embeddings + num_added_toks + simmc_added_tokens)

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
        train_dataset = SIMMCFineTuneDataset(train_raw_data, args.features_path, args, self.tokenizer, args.randomization)
        train_sampler = DistributedSampler(train_dataset) if args.distributed else Sampler(train_dataset)
        self.train_loader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      collate_fn=train_dataset.collate_fn)
    
        print('Building the val loader')
        val_raw_data = json.load(open(args.valid_path, 'r', encoding='utf-8'))
        val_dataset = SIMMCFineTuneDataset(val_raw_data, args.features_path, args, self.tokenizer, "no_random")
        self.val_loader = DataLoader(val_dataset,
                                    batch_size=args.valid_batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    sampler=None,
                                    collate_fn=val_dataset.collate_fn,
                                    drop_last=False)

        print('Building the test loader')
        test_raw_data = json.load(open(args.test_path, 'r', encoding='utf-8'))
        test_dataset = SIMMCFineTuneDataset(test_raw_data, args.features_path, args, self.tokenizer, "no_random")
        self.test_loader = DataLoader(test_dataset,
                                    batch_size=args.valid_batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    collate_fn=test_dataset.collate_fn,
                                    sampler=None,
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

        
        # Initialize evaluator
        self.evaluator = SIMMCEvaluator()
            
    def train(self):

        # If we just want to evalute
        if self.args.do_test:
            test_results = self.evaluate(self.test_loader, args.test_path, os.path.join(args.output, "test/"))

            return -1

        if self.verbose:
            loss_meter = LossMeter()
            best_valid = 0.
            best_epoch = 0

            if not self.wandb_initialized:
                if 't5' in self.args.backbone:
                    project_name = "VLT5_SIMMC"
                elif 'bart' in self.args.backbone:
                    project_name = "VLBart_SIMMC"

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
                pbar = tqdm(total=len(self.train_loader), ncols=120)

            epoch_results = {
                'loss': 0.,

            }

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
                        epoch_results[k] += v.item()

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
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step}'
                    desc_str += f' | Loss {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                pbar.close()

            # Validation
            valid_results = self.validate(self.val_loader)
            valid_metrics = self.evaluate(self.val_loader, args.valid_path, os.path.join(args.output, 'valid/'))

            if self.verbose:
                if self.args.optimize_ja:
                    try:
                        valid_score = 1.0 - valid_metrics['joint_accuracy']
                    except:
                        valid_score = 1.0
                else:
                    valid_score = valid_results['loss']

                if valid_score < best_valid or epoch == 0:
                    best_valid = valid_score
                    best_epoch = epoch
                    self.save("BEST")

                log_str = ''

                log_str += pformat(valid_results)
                log_str += "\nEpoch %d: Valid loss %0.4f" % (epoch, valid_score)
                log_str += "\nEpoch %d: Best loss %0.4f\n" % (best_epoch, best_valid)

                wandb_log_dict = {}
                wandb_log_dict['Train/Loss'] = epoch_results['loss'] / len(self.train_loader)

                for score_name, score in valid_results.items():
                    wandb_log_dict[f'Valid/{score_name}'] = score

                wandb_log_dict[f'Valid/best_epoch'] = best_epoch

                wandb.log(wandb_log_dict, step=epoch)

                print(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")

        # Test Set
        best_path = os.path.join(self.args.output, 'BEST')
        self.load(best_path)

        if self.verbose:
            wandb.save(best_path, base_path=self.args.output)
            print(f'\nUploaded checkpoint {best_epoch}', best_path)

        test_results = self.evaluate(self.test_loader, args.test_path, os.path.join(args.output, "test/"))

        if self.verbose:
            wandb_log_dict = {}
            for score_name, score in test_results.items():
                wandb_log_dict[f'Test/{score_name}'] = score
            wandb.log(wandb_log_dict, step=epoch)

            log_str = 'Test set results\n'
            log_str += pformat(test_results)

            print(log_str)

        if self.args.distributed:
            dist.barrier()


    def predict(self, loader):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = []
            examples = []

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            for i, batch in enumerate(tqdm(loader, ncols=120, desc="Prediction", disable=not self.verbose)):
                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                predictions.extend(results['pred'])

                if 'examples' in batch:
                    examples.extend(batch['examples'])
            results = {
                'predictions': predictions,
                'examples': examples
            }

            if self.args.distributed:
                dist.barrier()
                dist_results = dist_utils.all_gather(results)
                predictions = []
                examples = []
                for result in dist_results:
                    predictions.extend(result['predictions'])
                    examples.extend(result['examples'])
                results = {
                    'predictions': predictions,
                    'examples': examples
                }
            return results

    def validate(self, loader):
        self.model.eval()
        with torch.no_grad():
            losses = []
            for i, batch in enumerate(tqdm(loader, ncols=120, desc="Prediction", disable=not self.verbose)):
                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)
                losses.extend(results['loss'])

            results = {'loss': losses}
            
            if self.args.distributed:
                dist.barrier()
                dist_results = dist_utils.all_gather(results)
                losses = []
                for result in dist_results:
                    losses.extend(result['loss'])
            results = {
                'loss': np.mean(losses),
            }
            return results
        
    def evaluate(self, loader, gt_path, output_path):

        results = self.predict(loader)

        predictions = results['predictions']
        print('# predictions:', len(predictions))
        examples = results['examples']
        evaluator = self.evaluator
        eval_results = evaluator.evaluate(predictions, examples, gt_path, output_path)
        return eval_results

        
def main(gpu, args):

    args.gpu = gpu
    args.rank = gpu
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')


    if args.do_train:
        trainer = Trainer(args, train=True)
        trainer.train()
        

if __name__ == '__main__':

    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

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
