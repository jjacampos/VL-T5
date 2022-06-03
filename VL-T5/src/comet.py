from comet_data import COMETFineTuneDataset
import json
from utils import get_memories_mappings
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import h5py


from utils import load_state_dict, LossMeter, set_global_logging_level
from trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from comet_model import VLT5COMET, VLBartCOMET

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5COMET
        elif 'bart' in args.backbone:
            model_class = VLBartCOMET

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {
                    'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(
                    special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids(
                    [f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(
                self.model.model.shared.num_embeddings + num_added_toks)
            if self.verbose:
                print(f'Vocab resize: {self.tokenizer.vocab_size} -> {self.model.model.shared.num_embeddings}')
                assert self.model.model.shared.weight is self.model.lm_head.weight
                assert self.model.model.shared.weight is self.model.model.encoder.visual_embedding.obj_order_embedding.weight

        self.model.tokenizer = self.tokenizer
        if 't5' in self.args.tokenizer or 'bart' in self.args.tokenizer:
            self.model.true_id = self.tokenizer('true', add_special_tokens=False).input_ids[0]
            self.model.false_id = self.tokenizer('false', add_special_tokens=False).input_ids[0]

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

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
        # TODO


    def predict(self):
        # TODO
        
    



def main(args):

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    # Set the coco API and the mapping from memory ids to coco ids
    print('Getting mapping memories to coco ids')
    memories_to_coco_ids = get_memories_mappings(args)

    # Load the coco image features
    coco_features = h5py.File(args.coco_features_path, 'r')

    if args.do_train:
        print('Building the train loader')
        train_raw_data = json.load(open(args.train_path, 'r', encoding='utf-8'))
        train_dataset = COMETFineTuneDataset(train_raw_data, memories_to_coco_ids, coco_features, args)
        train_sampler = DistributedSampler(train_dataset) if args.distributed else Sampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      num_workers=args.workers,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      collate_fn=train_dataset.collate_fn)
    
        print('Building the val loader')
        val_raw_data = json.load(open(args.val_path, 'r', encoding='utf-8'))
        val_dataset = COMETFineTuneDataset(val_raw_data, memories_to_coco_ids, coco_features, args)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.test_batch_size,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=True,
                                    sampler=None,
                                    drop_last=False)

        trainer = Trainer(args, train_dataloader, val_dataset, train=True)
        trainer.train()
        
    if args.do_test:
        print('Building the test loader')
        test_raw_data = json.load(open(args.test_path, 'r', encoding='utf-8'))
        test_dataset = COMETFineTuneDataset(test_raw_data, memories_to_coco_ids, coco_features, args)
        test_dataloader = DataLoader(test_dataset,
                                    batch_size=args.test_batch_size,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=True,
                                    sampler=None,
                                    drop_last=False)
            

if __name__ == '__main__':
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    main(args)
    
