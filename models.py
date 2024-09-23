from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig, AutoProcessor

from triplet_mask import construct_mask


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


def build_custommm_model(args) -> nn.Module:
    return CustomMMModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


@dataclass
class ModelDoubleOutput:
    logits: torch.tensor
    logits_mm: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomMMModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.vit_config = AutoConfig.from_pretrained('./PLMs/vit-base-patch16-224')
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.6592),
                                              requires_grad=True)  # nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.bert = deepcopy(self.hr_bert)
        self.vit = AutoModel.from_pretrained('./PLMs/vit-base-patch16-224')
        self.prefix = self.args.prefix
        assert self.prefix > 0, "Set args.prefix > 0 to add visual prefix."
        self.vision_mapping = nn.Sequential(
            nn.Linear(self.vit_config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.config.hidden_size * self.prefix),
            nn.Sigmoid()
        )

    def _freeze_module(self, encoder, exclude: [str] = None):
        assert isinstance(encoder, torch.nn.Module), 'Argument must be nn.Module'
        for name, p in encoder.named_parameters():
            if exclude:
                for mod in exclude:
                    if mod in name:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            else:
                p.requires_grad = False

    def _encode_image(self, entity_pixel_values):
        img_outputs = self.vit(entity_pixel_values)
        last_hidden_state = img_outputs.last_hidden_state
        img_outputs = last_hidden_state[:, 0, :]
        img_embeds = img_outputs / img_outputs.norm(p=2, dim=-1, keepdim=True)
        img_embeds = self.vision_mapping(img_embeds).reshape(-1, self.prefix, self.config.hidden_size)
        return img_embeds

    def _encode_text(self, encoder, input_ids=None, attention_mask=None, inputs_embeds=None, seq_output=None):
        outputs = encoder(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=None,
                          inputs_embeds=inputs_embeds,
                          return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        if seq_output:
            return last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        text_outputs = _pool_output(self.args.pooling, cls_output, attention_mask, last_hidden_state)
        text_embeds = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

    def emb_with_prefix(self, token_ids, mask, img_feature, seq_output=None):
        input_embeds = self.bert.embeddings(input_ids=token_ids)  # bs, seq_len, dim
        input_embeds[:, 1:1 + self.prefix, :] = img_feature  # bs, seq_len, dim
        prefixed_emb = self._encode_text(encoder=self.bert, attention_mask=mask,
                                         inputs_embeds=input_embeds, seq_output=seq_output)
        return prefixed_emb

    def forward(self, hr_token_ids, hr_mask, tr_token_ids, tr_mask,
                head_pixel_values, tail_pixel_values,
                head_token_ids, head_mask,
                tail_token_ids, tail_mask,
                # hr_token_ids, hr_mask, hr_token_type_ids,
                # tail_token_ids, tail_mask, tail_token_type_ids,
                # head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:  # in predict.py predict_by_entities
            return self.predict_ent_embedding(pixel_values=tail_pixel_values,
                                              token_ids=tail_token_ids, attention_mask=tail_mask)

        hr_embeds = self._encode_text(self.hr_bert, hr_token_ids, hr_mask)
        tail_embeds = self._encode_image(tail_pixel_values)
        head_embeds = self._encode_image(head_pixel_values)
        head_mm_embeds = self.emb_with_prefix(token_ids=head_token_ids, mask=head_mask, img_feature=head_embeds)
        tail_mm_embeds = self.emb_with_prefix(token_ids=tail_token_ids, mask=tail_mask, img_feature=tail_embeds)
        # hr_embeds = self.emb_with_prefix(token_ids=hr_token_ids, mask=hr_mask, img_feature=head_embeds)  # hr + vis
        return {
            'hr_vector': hr_embeds,
            'tail_vector': tail_embeds.mean(dim=1).float(),
            'head_vector': head_embeds.mean(dim=1).float(),
            'tail_mm_vector': tail_mm_embeds,
            'head_mm_vector': head_mm_embeds,
        }

    def _compute_neg_logits(self, logits, hr_vector, tail_vector, head_vector, batch_dict, logit_scale):
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * logit_scale * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        return logits

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector, head_vector = output_dict['hr_vector'], \
            output_dict['tail_vector'], output_dict['head_vector']

        # hr_vector, tail_vector, tr_vector, head_vector = output_dict.values()
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)
        logit_scale = self.logit_scale.exp()
        hr_logits = hr_vector.mm(tail_vector.t()) * logit_scale
        hr_logits = self._compute_neg_logits(hr_logits, hr_vector, tail_vector, head_vector, batch_dict, logit_scale)
        tail_mm_vector, head_mm_vector = output_dict['tail_mm_vector'], output_dict['head_mm_vector']
        hr_logits_mm = hr_vector.mm(tail_mm_vector.t()) * logit_scale
        hr_logits_mm = self._compute_neg_logits(hr_logits_mm, hr_vector,
                                                tail_mm_vector, head_mm_vector, batch_dict, logit_scale)
        return {'logits': hr_logits,
                'logits_mm': hr_logits_mm,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:

        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        logit_scale = self.logit_scale.exp()
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t()) * logit_scale
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, pixel_values, token_ids=None, attention_mask=None, **kwargs) -> dict:
        ent_embeds = self._encode_image(pixel_values)
        if token_ids is None or attention_mask is None:
            return {'ent_vectors': ent_embeds.detach()}
        ent_prefix_embeds = self.emb_with_prefix(token_ids=token_ids, mask=attention_mask, img_feature=ent_embeds,
                                                 seq_output=False)
        return {
            'ent_vectors': ent_embeds.mean(dim=1).float(),
            'ent_vectors_mm': ent_prefix_embeds.detach()
        }


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(),
                                            requires_grad=args.finetune_t)  # log(1/temp)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,  # [bs, hs]
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)  # [bs]

        logits = hr_vector.mm(tail_vector.t())  # [bs, bs]
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()  # [bs, bs]

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)  # [bs, 2*bs]
            logits = torch.cat([logits, pre_batch_logits], dim=-1)  # [bs, bs+2*bs]

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)  # [bs, bs+2*bs+1]

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())  # [bs, 2*bs]
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
