import os
import json
import tqdm
import torch
import torch.utils.data

from typing import List
from collections import OrderedDict

from doc import collate, Example, Dataset
from config import args
from models import build_model, build_custommm_model
from utils import AttrDict, move_to_cuda
from dict_hub import build_tokenizer, build_processor
from logger_config import logger
from models import ModelOutput


class BertPredictor:

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False

    def load(self, ckt_path, use_data_parallel=False):
        assert os.path.exists(ckt_path)
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        if args.mm:
            build_processor(self.train_args)
        else:
            build_tokenizer(self.train_args)
        if args.mm:
            self.model = build_custommm_model(self.train_args)
            logger.info("Build CustomMM Model.")
        else:
            self.model = build_model(self.train_args)
            logger.info("Build CustomBertModel"
                        "All paras in both hr_bert and tail_bert requires_grad=True")
        # self.model = build_model(self.train_args)

        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        if args.adapter_model:
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Use data parallel predictor')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            self.model.cuda()
            self.use_cuda = True
        logger.info('Load model from {} successfully'.format(ckt_path))

    def _setup_args(self):
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                logger.info('Set default attribute: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        logger.info(
            'Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        args.use_link_graph = self.train_args.use_link_graph
        args.is_test = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]):
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=1,
            batch_size=max(args.batch_size, 512),
            collate_fn=collate,
            shuffle=False)

        hr_tensor_list, tail_tensor_list = [], []
        logger.info("Start to predict by examples. ")
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader, ncols=80)):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
            tail_tensor_list.append(outputs['tail_vector'])

        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id=entity_ex.entity_id, relation='',
                                    # head_id in entity_ex not used in only_ent_embedding
                                    tail_id=entity_ex.entity_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=2,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)

        ent_tensor_list = []
        ent_tensor_mm_list = []
        logger.info("Start to predict by entities. ")
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader, ncols=80)):
            batch_dict['only_ent_embedding'] = True  # only return entity embeddings
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])
            if args.mm:
                ent_tensor_mm_list.append(outputs['ent_vectors_mm'])

        return torch.cat(ent_tensor_list, dim=0), \
            torch.cat(ent_tensor_mm_list, dim=0) if ent_tensor_mm_list else None
