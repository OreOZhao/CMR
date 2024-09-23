import os
import json
import tqdm
import torch

from time import time
from typing import List, Tuple
from dataclasses import dataclass, asdict

from config import args
from doc import load_data, Example
from predict import BertPredictor
from dict_hub import get_entity_dict, get_all_triplet_dict
from triplet import EntityDict
from logger_config import logger
import pickle
from knowledge_store import BertKnowledgeStore
import random

args.seed = 202303
random.seed(args.seed)
torch.manual_seed(args.seed)


def _setup_entity_dict() -> EntityDict:
    return get_entity_dict()


entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()

knn_lambda_list = [i * 0.025 for i in range(int(1.026 / 0.025))]
logger.info("lambda_list {}".format(knn_lambda_list))


@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool


@torch.no_grad()
def compute_metrics(hr_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target: List[int],
                    examples: List[Example],
                    k=3, batch_size=256, predictor: BertKnowledgeStore = None) -> Tuple:
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0

    for start in tqdm.tqdm(range(0, total, batch_size), ncols=80):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        # filter known triplets
        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks


@torch.no_grad()
def compute_metrics_search_knn_lamda(hr_tensor: torch.tensor,
                                     entities_tensor: torch.tensor,
                                     target: List[int],
                                     examples: List[Example],
                                     k=3, batch_size=256, predictor: BertKnowledgeStore = None) -> Tuple:
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = [0] * len(knn_lambda_list), [0] * len(knn_lambda_list), \
                                        [0] * len(knn_lambda_list), [0] * len(knn_lambda_list), \
                                        [0] * len(knn_lambda_list)

    for start in tqdm.tqdm(range(0, total, batch_size), ncols=80):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]
        batch_exes = examples[start:end]
        # batch_score = torch.nn.functional.softmax(batch_score, dim=1)

        batch_score_list = []

        if args.knn_topk > 0:
            batch_score_knn = predictor.batch_prob_knn_filter_target(hr_tensor[start:end, :], batch_score, batch_exes)
            assert batch_score.shape == batch_score_knn.shape
            for l in knn_lambda_list:
                bscore = l * batch_score_knn + (1. - l) * batch_score
                batch_score_list.append(bscore)

        for i in range(len(batch_score_list)):
            batch_score = batch_score_list[i]
            for idx in range(batch_score.size(0)):
                mask_indices = []
                cur_ex = examples[start + idx]
                gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
                if len(gold_neighbor_ids) > 10000:
                    logger.debug(
                        '{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
                for e_id in gold_neighbor_ids:
                    if e_id == cur_ex.tail_id:
                        continue
                    mask_indices.append(entity_dict.entity_to_idx(e_id))
                mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
                batch_score[idx].index_fill_(0, mask_indices, -1)

            batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
            target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
            assert target_rank.size(0) == batch_score.size(0)
            for idx in range(batch_score.size(0)):
                idx_rank = target_rank[idx].tolist()
                assert idx_rank[0] == idx
                cur_rank = idx_rank[1]

                # 0-based -> 1-based
                cur_rank += 1
                mean_rank[i] += cur_rank
                mrr[i] += 1.0 / cur_rank
                hit1[i] += 1 if cur_rank <= 1 else 0
                hit3[i] += 1 if cur_rank <= 3 else 0
                hit10[i] += 1 if cur_rank <= 10 else 0
                ranks.append(cur_rank)

            topk_scores.extend(batch_sorted_score[:, :k].tolist())
            topk_indices.extend(batch_sorted_indices[:, :k].tolist())
    lambda_metric_list = []
    for i in range(len(knn_lambda_list)):
        metrics = {'mean_rank': mean_rank[i], 'mrr': mrr[i], 'hit@1': hit1[i], 'hit@3': hit3[i], 'hit@10': hit10[i]}
        metrics = {k: round(v / total, 4) for k, v in metrics.items()}
        lambda_metric_list.append(metrics)

    # assert len(topk_scores) == total
    return topk_scores, topk_indices, lambda_metric_list, ranks


def predict_by_mm_and_img():
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)

    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)
    entity_tensor, entity_tensor_mm = predictor.predict_by_entities(
        entity_dict.entity_exs)  # obtain entity embeddings

    forward_metrics_mm = eval_single_direction(predictor,
                                               entity_tensor=entity_tensor_mm,
                                               eval_forward=True)
    backward_metrics_mm = eval_single_direction(predictor,
                                                entity_tensor=entity_tensor_mm,
                                                eval_forward=False)
    metrics_mm = {k: round((forward_metrics_mm[k] + backward_metrics_mm[k]) / 2, 4)
                  for k in forward_metrics_mm}
    logger.info('Averaged metrics mm: {}'.format(metrics_mm))

    forward_metrics = eval_single_direction(predictor,
                                            entity_tensor=entity_tensor,
                                            eval_forward=True)
    backward_metrics = eval_single_direction(predictor,
                                             entity_tensor=entity_tensor,
                                             eval_forward=False)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))
        writer.write('forward metrics text: {}\n'.format(json.dumps(forward_metrics_mm)))
        writer.write('backward metrics text: {}\n'.format(json.dumps(backward_metrics_mm)))
        writer.write('average metrics text: {}\n'.format(json.dumps(metrics_mm)))


def eval_single_direction(predictor: BertPredictor,
                          entity_tensor: torch.tensor,
                          eval_forward=True,
                          batch_size=256) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)
    hr_tensor, _ = predictor.predict_by_examples(examples)  # hr_vector in forward()
    hr_tensor = hr_tensor.to(entity_tensor.device)
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')

    if args.knn_topk > 0:
        topk_scores, topk_indices, metrics, ranks = compute_metrics_search_knn_lamda(hr_tensor=hr_tensor,
                                                                                     entities_tensor=entity_tensor,
                                                                                     target=target, examples=examples,
                                                                                     batch_size=batch_size,
                                                                                     predictor=predictor)
    else:
        topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                                                                    target=target, examples=examples,
                                                                    batch_size=batch_size, predictor=predictor)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    pred_infos = []
    for idx, ex in enumerate(examples):
        cur_topk_scores = topk_scores[idx]
        cur_topk_indices = topk_indices[idx]
        pred_idx = cur_topk_indices[0]
        cur_score_info = {entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
                          for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices)}

        pred_info = PredInfo(head=ex.head, relation=ex.relation,
                             tail=ex.tail, pred_tail=entity_dict.get_entity_by_idx(pred_idx).entity,
                             pred_score=round(cur_topk_scores[0], 4),
                             topk_score_info=json.dumps(cur_score_info),
                             rank=ranks[idx],
                             correct=pred_idx == target[idx])
        pred_infos.append(pred_info)

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics


def predict_with_knowledge_store():
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)
    logger.info('knn k: {}'.format(args.knn_topk))
    predictor = BertKnowledgeStore()
    predictor.load(ckt_path=args.eval_model_path)
    model_dir = os.path.dirname(args.eval_model_path)
    entity_tensor, entity_tensor_mm = predictor.predict_by_entities(
        entity_dict.entity_exs)  # obtain entity embeddings
    # if not os.path.exists(os.path.join(model_dir, "entity_tensor.pickle")):
    #     pickle.dump(entity_tensor, open(os.path.join(model_dir, "entity_tensor.pickle"), 'wb'))
    predictor.faiss_index_init()

    if entity_tensor_mm is not None:
        forward_metrics_mm = eval_single_direction(predictor,
                                                   entity_tensor=entity_tensor_mm,
                                                   eval_forward=True)
    backward_metrics_mm = eval_single_direction(predictor,
                                                entity_tensor=entity_tensor_mm,
                                                eval_forward=False)

    logger.info('knn k: {}'.format(args.knn_topk))
    for i in range(len(knn_lambda_list)):
        metrics_mm = {k: round((forward_metrics_mm[i][k] + backward_metrics_mm[i][k]) / 2, 4)
                      for k in forward_metrics_mm[i]}
        logger.info('knn lambda: {}'.format(knn_lambda_list[i]))
        logger.info('Averaged metrics mm: {}'.format(metrics_mm))

    forward_metrics = eval_single_direction(predictor,
                                            entity_tensor=entity_tensor,
                                            eval_forward=True)
    backward_metrics = eval_single_direction(predictor,
                                             entity_tensor=entity_tensor,
                                             eval_forward=False)
    logger.info('knn k: {}'.format(args.knn_topk))
    for i in range(len(knn_lambda_list)):
        metrics = {k: round((forward_metrics[i][k] + backward_metrics[i][k]) / 2, 4)
                   for k in forward_metrics[i]}
        logger.info('knn lambda: {}'.format(knn_lambda_list[i]))
        logger.info('Averaged metrics: {}'.format(metrics))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))
        writer.write('forward metrics text: {}\n'.format(json.dumps(forward_metrics_mm)))
        writer.write('backward metrics text: {}\n'.format(json.dumps(backward_metrics_mm)))
        writer.write('average metrics text: {}\n'.format(json.dumps(metrics_mm)))


if __name__ == '__main__':
    if args.knn_topk > 0:
        predict_with_knowledge_store()
    else:
        predict_by_mm_and_img()
