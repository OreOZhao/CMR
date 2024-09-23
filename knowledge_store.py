import os
import json
import tqdm
import torch
import torch.utils.data

from typing import List
from collections import OrderedDict

from doc import collate, Example, Dataset, load_data
from config import args
from models import build_model, build_custommm_model
from utils import AttrDict, move_to_cuda
from dict_hub import build_tokenizer, build_processor
from logger_config import logger
from models import ModelOutput
import faiss
from dict_hub import get_entity_dict
from predict import BertPredictor
import pickle


class BertKnowledgeStore(BertPredictor):

    def __init__(self):
        super().__init__()
        self.faissid2entityid = {}
        self.ent_faiss2entityid = {}
        self.entity_dict = get_entity_dict()

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
            self.dim = self.model.config.hidden_size  # CustomMMModel
            self.index = faiss.IndexFlatL2(self.dim)
            self.ent_index = faiss.IndexFlatL2(self.dim)
            logger.info("Build IndexFlatL2 dim: {}".format(self.dim))
        else:
            self.model = build_model(self.train_args)
            logger.info("Build CustomBertModel"
                        "All paras in both hr_bert and tail_bert requires_grad=True")
            self.dim = self.model.config.hidden_size
            self.index = faiss.IndexFlatL2(self.dim)
            logger.info("Build IndexFlatL2 dim: ", self.dim)
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
    def _entity_predict_by_examples(self, examples: List[Example]):
        bsz = max(args.batch_size, 512)

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=1,
            batch_size=bsz,
            collate_fn=collate,
            shuffle=False)

        hr_tensor_list, tail_tensor_list, tail_mm_vector_list, tail_entity_ids = [], [], [], []
        logger.info("Start to initialize faiss index vectors. ")
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader, ncols=80)):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
            tail_tensor_list.append(outputs['tail_vector'])
            if 'tail_mm_vector' in outputs:
                tail_mm_vector_list.append(outputs['tail_mm_vector'])
            batch_tail_ent_ids = [batch_dict['batch_data'][i].tail_ent_id for i in range(len(batch_dict['batch_data']))]
            tail_entity_ids.append(batch_tail_ent_ids)
            for i in range(len(batch_tail_ent_ids)):
                self.faissid2entityid[idx * bsz + i] = batch_tail_ent_ids[i]

        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0), \
            torch.cat(tail_mm_vector_list, dim=0) if tail_mm_vector_list else None, tail_entity_ids

    def faiss_index_init(self):
        model_dir = os.path.dirname(args.eval_model_path)
        index_file = 'hr_index.index'
        f2eid_file = 'faissid2entityid.pickle'
        if os.path.exists(os.path.join(model_dir, index_file)):
            validation_set_path = args.valid_path.replace('test', 'valid')
            train_examples = load_data(args.train_path, add_forward_triplet=True, add_backward_triplet=True) + \
                             load_data(validation_set_path, add_forward_triplet=True, add_backward_triplet=True)
            if args.task.endswith('_ind'):  # test graph of inductive KGC
                train_examples += load_data(args.valid_path, add_forward_triplet=True, add_backward_triplet=True)
            self.train_examples = train_examples
            self.index = faiss.read_index(os.path.join(model_dir, index_file))
            self.faissid2entityid = pickle.load(open(os.path.join(model_dir, f2eid_file), 'rb'))
            logger.info("Loaded index from .index file. ")

        else:
            validation_set_path = args.valid_path.replace('test', 'valid')
            train_examples = load_data(args.train_path, add_forward_triplet=True, add_backward_triplet=True) + \
                             load_data(validation_set_path, add_forward_triplet=True, add_backward_triplet=True)
            if args.task.endswith('_ind'):
                train_examples += load_data(args.valid_path, add_forward_triplet=True, add_backward_triplet=True)
            self.train_examples = train_examples
            hr_vector, tail_vector, tail_mm_vector, tail_entity_ids = self._entity_predict_by_examples(train_examples)
            hr_numpy = hr_vector.cpu().numpy()
            print(faiss.MatrixStats(hr_numpy).comments)
            self.index.add(hr_numpy)
            pickle.dump(self.faissid2entityid, open(os.path.join(model_dir, f2eid_file), 'wb'))
            faiss.write_index(self.index, os.path.join(model_dir, index_file))

    def batch_prob_knn_filter_target(self, batch_hr_tensor, batch_score, batch_target):
        topk = args.knn_topk
        bsz = batch_hr_tensor.shape[0]

        batch_hr_numpy = batch_hr_tensor.cpu().numpy()
        D, I = self.index.search(batch_hr_numpy, topk)  # I: ids [bs, topk]

        entity_logits = torch.full(batch_score.shape, 1000.).to(batch_hr_tensor.device)

        D = torch.from_numpy(D).to(batch_score.device)

        for sample in range(bsz):
            for neighbor in range(topk):
                ent_id = self.faissid2entityid[I[sample][neighbor]]
                retrieved_query_id = I[sample][neighbor]
                if retrieved_query_id >= len(self.train_examples):
                    continue
                retrieved_query = self.train_examples[retrieved_query_id]
                cur_ex = batch_target[sample]
                if cur_ex.head_ent_id == retrieved_query.head_ent_id and cur_ex.rel_id == retrieved_query.rel_id:
                    continue
                if entity_logits[sample][ent_id] == 1000.:
                    entity_logits[sample][ent_id] = D[sample][neighbor]

        entity_logits = distance2logits_2(entity_logits)  # or last softmax

        return entity_logits

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


def distance2logits_2(D, n=10):
    if not isinstance(D, torch.Tensor):
        D = torch.tensor(D)
    # if torch.sum(D) != 0.0:
    D = torch.exp(-D / n) / torch.sum(torch.exp(-D / n), dim=-1, keepdim=True)
    return D
