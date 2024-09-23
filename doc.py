import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer, get_processor, get_all_triplet_dict
from logger_config import logger
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

entity_dict = get_entity_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()

all_triplet_dict = get_all_triplet_dict()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _custom_mm_processor(prompt: str, image_path: str) -> dict:
    processor = get_processor()
    image = Image.open(image_path).convert('RGB').resize((384, 384))
    if args.mm:
        img_inputs = processor(images=image,
                               return_tensors='pt')
        text_inputs = _custom_tokenize(text=prompt)
        encoded_inputs = {'input_ids': text_inputs['input_ids'],
                          'pixel_values': img_inputs['pixel_values']}
    else:
        encoded_inputs = processor(text=prompt,
                                   images=image,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True)
    return encoded_inputs


def _custom_mm_tokenizer(text: str) -> dict:
    processor = get_processor()
    encoded_inputs = processor.tokenizer(text=text,
                                         return_tensors="pt",
                                         padding=True,
                                         truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head_img(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_imgpath

    @property
    def tail_img(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_imgpath

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    @property
    def rel_id(self):
        return all_triplet_dict.rel2id.get(self.relation, -1)

    @property
    def head_ent_id(self):
        return entity_dict.entity_to_idx(self.head_id)

    @property
    def tail_ent_id(self):
        return entity_dict.entity_to_idx(self.tail_id)

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        head_imgpath, tail_imgpath = self.head_img, self.tail_img
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        tail_word = _parse_entity_name(self.tail)
        tail_text = _concat_name_desc(tail_word, tail_desc)
        if args.mm:
            hr_prompt = 'A photo of ' + head_word + "'s " + self.relation + '. ' + head_desc
            tr_prompt = 'A photo of entity with ' + self.relation + ' as ' + tail_word + '. ' + tail_desc
            if args.prefix > 0:
                hr_prompt = 'x ' * args.prefix + hr_prompt
                tr_prompt = 'x ' * args.prefix + tr_prompt
            hr_mm_inputs = _custom_mm_processor(hr_prompt, tail_imgpath)
            tr_mm_inputs = _custom_mm_processor(tr_prompt, head_imgpath)
            if args.mm:
                if args.prefix > 0:
                    head_text = 'x ' * args.prefix + head_text
                    tail_text = 'x ' * args.prefix + tail_text
                head_encoded_inputs = _custom_tokenize(head_text)
                tail_encoded_inputs = _custom_tokenize(tail_text)
            else:
                head_encoded_inputs = _custom_mm_tokenizer(head_text)
                tail_encoded_inputs = _custom_mm_tokenizer(tail_text)
            return {'hr_token_ids': hr_mm_inputs['input_ids'],
                    'tail_pixel_values': hr_mm_inputs['pixel_values'],
                    'tr_token_ids': tr_mm_inputs['input_ids'],
                    'head_pixel_values': tr_mm_inputs['pixel_values'],
                    'head_token_ids': head_encoded_inputs['input_ids'],
                    'tail_token_ids': tail_encoded_inputs['input_ids'],
                    'obj': self}
        else:
            hr_encoded_inputs = _custom_tokenize(text=head_text,
                                                 text_pair=self.relation)

            head_encoded_inputs = _custom_tokenize(text=head_text)

            tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

            return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                    'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                    'tail_token_ids': tail_encoded_inputs['input_ids'],
                    'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                    'head_token_ids': head_encoded_inputs['input_ids'],
                    'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                    'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    if args.mm:
        pad_token_id = get_tokenizer().pad_token_id if args.mm else get_processor().tokenizer.pad_token_id
        hr_token_ids, hr_mask = to_indices_and_mask(
            [torch.LongTensor(ex['hr_token_ids']).squeeze() for ex in batch_data],
            pad_token_id=pad_token_id)
        tr_token_ids, tr_mask = to_indices_and_mask(
            [torch.LongTensor(ex['tr_token_ids']).squeeze() for ex in batch_data],
            pad_token_id=pad_token_id)

        head_token_ids, head_mask = to_indices_and_mask(
            [torch.LongTensor(ex['head_token_ids']).squeeze() for ex in batch_data],
            pad_token_id=pad_token_id)
        tail_token_ids, tail_mask = to_indices_and_mask(
            [torch.LongTensor(ex['tail_token_ids']).squeeze() for ex in batch_data],
            pad_token_id=pad_token_id)

        head_pixel_values = torch.cat([torch.FloatTensor(ex['head_pixel_values']) for ex in batch_data], dim=0)
        tail_pixel_values = torch.cat([torch.FloatTensor(ex['tail_pixel_values']) for ex in batch_data], dim=0)

        batch_exs = [ex['obj'] for ex in batch_data]
        batch_dict = {
            'hr_token_ids': hr_token_ids,
            'hr_mask': hr_mask,
            'tr_token_ids': tr_token_ids,
            'tr_mask': tr_mask,
            'head_pixel_values': head_pixel_values,
            'tail_pixel_values': tail_pixel_values,
            'head_token_ids': head_token_ids,
            'head_mask': head_mask,
            'tail_token_ids': tail_token_ids,
            'tail_mask': tail_mask,
            'batch_data': batch_exs,
            'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,  # (bs, bs)
            'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        }
        return batch_dict
    else:
        hr_token_ids, hr_mask = to_indices_and_mask(
            [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
            pad_token_id=get_tokenizer().pad_token_id)
        hr_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
            need_mask=False)

        tail_token_ids, tail_mask = to_indices_and_mask(
            [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
            pad_token_id=get_tokenizer().pad_token_id)
        tail_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
            need_mask=False)

        head_token_ids, head_mask = to_indices_and_mask(
            [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
            pad_token_id=get_tokenizer().pad_token_id)
        head_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
            need_mask=False)

        batch_exs = [ex['obj'] for ex in batch_data]
        batch_tensor = torch.tensor([(ex.head_ent_id, ex.rel_id, ex.tail_ent_id) for ex in batch_exs])
        batch_dict = {
            'hr_token_ids': hr_token_ids,
            'hr_mask': hr_mask,
            'hr_token_type_ids': hr_token_type_ids,
            'tail_token_ids': tail_token_ids,
            'tail_mask': tail_mask,
            'tail_token_type_ids': tail_token_type_ids,
            'head_token_ids': head_token_ids,
            'head_mask': head_mask,
            'head_token_type_ids': head_token_type_ids,
            'batch_tensor': batch_tensor,
            'batch_data': batch_exs,
            'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
            'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        }

        return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
