import math
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import random

try:
    from util.topk import top_k
    from .hico_text_label import hico_text_label
except ModuleNotFoundError:
    from topk import top_k
    from hico_text_label import hico_text_label

try:
    from .triplet_loss import batch_hard_triplet_loss
except ImportError:
    from triplet_loss import batch_hard_triplet_loss


def apply_risk_mask(choice_list, mask, target):
    if len(choice_list) > 1:
        choice_list = torch.cat(choice_list)
        verb = choice_list[torch.randint(len(choice_list) - 1, size=(100,))]
        target[mask] = verb[mask]
    elif len(choice_list) == 1:
        choice_list = torch.cat(choice_list)
        verb = torch.full((100,), choice_list[0])
        target[mask] = verb[mask]
    return target


class SentenceCriterion:
    def __init__(self,
                 # embedding_file="./checkpoint/hoi_clip_embedding.pth",
                 embedding_file="./checkpoint/synth_hoi_clip_embedding_ckpt.pth",
                 # embedding_file="./checkpoint/pari_choice_clip_embedding_ckpt.pth",
                 clip_model="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.device = device

        # pair example: (action_id, object_id)
        self.pair2text = hico_text_label
        self.text2pair = {v: k for k, v in hico_text_label.items()}

        # special sentence
        self.special_sentence = "A photo of a person."
        if not os.path.exists("./checkpoint/special_sentence_tensor.pth"):
            # tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            # model_proj = CLIPTextModelWithProjection.from_pretrained(clip_model)
            # inputs = tokenizer([self.special_sentence], padding=True, return_tensors="pt", max_length=13)
            # outputs = model_proj(**inputs)
            # # print(outputs.text_embeds, outputs.text_embeds.size())
            # torch.save({"A photo of a person.": outputs.text_embeds}, "./checkpoint/special_sentence_tensor.pth")
            # self.special_sentence_tensor = outputs.text_embeds
            pass
        else:
            self.special_sentence_tensor = torch.load("./checkpoint/special_sentence_tensor.pth")[self.special_sentence]
            print("load special sentence tensor from pretrained.", self.special_sentence_tensor.size())
            # self.special_sentence_tensor

        self.verb2text = {}
        self.text2verb = {}

        self.l1_loss = nn.L1Loss()
        # When using pretrained weight "./checkpoint/hoi_clip_embedding.pth"

        # self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        # When using pretrained weight "./checkpoint/hoi_clip_embedding_log_space.pth"

        # label composition
        self.p_v = 0.6  # 0.6 to change, 0.4 to not change

        self.p_verb = 0.8  # 0.2 to change verb, 0.8 to not change
        self.p_obj = 0.2  # 0.8 to change verb, 0.2 to not change

        self.k = 30

        if embedding_file is not None:
            print("Load hoi_clip_embedding from pretrained.")
            self.text2tensor = torch.load(embedding_file)
            # self.text_embeddings = torch.cat(list(self.text2tensor.values()))  # torch.Size([600, 512])

        else:
            # text2tensor = {}
            # tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            # model_proj = CLIPTextModelWithProjection.from_pretrained(clip_model)
            # print("hoi_clip_embedding.pth is not given, using clip model to encoding.")
            # for i, text in enumerate(self.text2pair.keys()):
            #     inputs = tokenizer([text], padding=True, return_tensors="pt", max_length=13)
            #     outputs = model_proj(**inputs)
            #     text2tensor[text] = F.log_softmax(outputs.text_embeds, dim=1)
            # self.text2tensor = text2tensor
            # torch.save(text2tensor, "./checkpoint/hoi_clip_embedding_log_space.pth")
            pass

        if "ckpt" in embedding_file:
            ckpt = torch.load(embedding_file)
            self.text_embeddings = ckpt["text_tensor"]
            self.text2idx = ckpt["sentence2tensor_id"]
            self.pair2tensor_id = ckpt["pair2tensor_id"]
            self.pair2tensor_id_narrow = {k: v for k, v in self.pair2tensor_id.items() if k in hico_text_label}
            print(f"Len of pair2tensor_id_narrow {len(self.pair2tensor_id_narrow)}")
            self.idx2text = {v: k for k, v in self.text2idx.items()}
            self.real2synth_tensor_id = ckpt['real2synth_tensor_id']
            self.real2synth_tensor_id = {k: torch.tensor(v) for k, v in self.real2synth_tensor_id.items()}

            # embedding_file="./checkpoint/pari_choice_clip_embedding_ckpt.pth",
            # self.pair_choice_tensor_id = ckpt['pair_choice_tensor_id']

        else:
            self.text_embeddings = torch.cat(list(self.text2tensor.values()))  # torch.Size([600, 512])
            self.list_text2tensor = list(self.text2tensor)
            self.text2idx = {text: self.list_text2tensor.index(text) for text in self.text2pair.keys()}
            self.idx2text = {v: k for k, v in self.text2idx.items()}
            self.real2synth_tensor_id = None

        self.text_embedding_norm = F.normalize(self.text_embeddings, dim=-1)
        sim_matrix = torch.mm(self.text_embedding_norm, self.text_embedding_norm.t())

        print(f"Similarity Matrix Size: {sim_matrix.size()}")

        top_k_result = torch.topk(sim_matrix, k=self.k, dim=-1)
        self.top_k_index = top_k_result.indices
        self.top_k_score = top_k_result.values

        # self.max_hoi = 100

    def batch_l1_loss(self, outputs, targets):
        """
        :param outputs: dict {
            'pred_sub_boxes': torch.Size([3, 64, 4]) = human_pred_boxes,
            'pred_obj_logits': torch.Size([3, 64, 81]) = [batch_size, num_queries, num_classes],
            'pred_obj_boxes': torch.Size([3, 64, 4]) = object_pred_boxes,
            'pred_hoi_logits': torch.Size([3, 64, 600]) = [batch_size, num_queries, classes],
            'action_pred_logits': action_outputs_class[-1],
            'pred_hoi_embedding': torch.Size([batch_size, num_query, 512])
        }
        :param targets: List[dict] {
            'hoi_sentence: ["a photo of a person lassoing a cow", ...]
        }
        :return losses: dict {
            'l1_loss': tensor
        }
        """
        assert 'pred_hoi_embeddings' in outputs
        # assert 'hoi_sentence' in targets
        pred_hoi_embeddings = outputs['pred_hoi_embeddings']
        device = pred_hoi_embeddings.device
        losses = {}

        # label composition
        collate_hoi_embeddings = []
        for i, t in enumerate(targets):
            text = t['hoi_sentence'][0]
            if text == self.special_sentence:
                collate_query_hoi_embeddings = self.special_sentence_tensor.repeat(1, 100, 1)
            else:
                embedding_idx = self.text2idx[text]
                top_k_indices = self.top_k_index[embedding_idx]

                mask = torch.rand(100) > self.p_v
                rand_idxs = torch.randint(1, len(top_k_indices), size=(1, 100))[0]
                rand_idxs[mask] = 0  # keep original embedding
                rand_similar_label = top_k_indices[rand_idxs]

                collate_query_hoi_embeddings = self.text_embeddings[rand_similar_label].unsqueeze(0)

            collate_hoi_embeddings.append(collate_query_hoi_embeddings)
        collate_hoi_embeddings = torch.cat(collate_hoi_embeddings, dim=0)
        losses['l1_loss'] = self.l1_loss(pred_hoi_embeddings, collate_hoi_embeddings.to(device))

        return losses

    def batch_l1_con_loss(self, outputs, targets):
        """
        :param outputs: dict {
            'pred_sub_boxes': torch.Size([3, 64, 4]) = human_pred_boxes,
            'pred_obj_logits': torch.Size([3, 64, 81]) = [batch_size, num_queries, num_classes],
            'pred_obj_boxes': torch.Size([3, 64, 4]) = object_pred_boxes,
            'pred_hoi_logits': torch.Size([3, 64, 600]) = [batch_size, num_queries, classes],
            'action_pred_logits': action_outputs_class[-1],
            'pred_hoi_embedding': torch.Size([batch_size, num_query, 512])
        }
        :param targets: List[dict] {
            'hoi_sentence: ["a photo of a person lassoing a cow", ...]
        }
        :return losses: dict {
            'l1_loss': tensor
        }
        """
        assert 'pred_hoi_embeddings' in outputs
        # assert 'hoi_sentence' in targets
        pred_hoi_embeddings = outputs['pred_hoi_embeddings']
        device = pred_hoi_embeddings.device
        losses = {}

        # label composition
        collate_hoi_embeddings = []
        for i, t in enumerate(targets):
            # text = t['hoi_sentence'][0]
            text = random.choice(t['hoi_sentence'])
            if text == self.special_sentence:
                # collate_query_hoi_embeddings = [self.special_sentence_tensor for i in range(100)]
                collate_query_hoi_embeddings = self.special_sentence_tensor.repeat(1, 100, 1)
                # print(collate_query_hoi_embeddings.size())
            else:
                embedding_idx = self.text2idx[text]
                # top_k_indices = self.top_k_index[embedding_idx]
                # print(embedding_idx)
                top_k_indices = self.real2synth_tensor_id[embedding_idx]
                # print(top_k_indices)
                # tensor([101, 104, 103,  98,  96, 105, 106,  97, 100, 102]) first one is original, e.g. 101

                mask = torch.rand(100) > self.p_v
                rand_idxs = torch.randint(1, len(top_k_indices), size=(100,))
                rand_idxs[mask] = 0  # keep original embedding
                rand_similar_label = top_k_indices[rand_idxs]

                collate_query_hoi_embeddings = self.text_embeddings[rand_similar_label].unsqueeze(0)

            collate_hoi_embeddings.append(collate_query_hoi_embeddings)

        collate_hoi_embeddings = torch.cat(collate_hoi_embeddings, dim=0)
        losses['l1_loss'] = self.l1_loss(pred_hoi_embeddings, collate_hoi_embeddings.to(device))

        return losses

    def batch_l1_triplet_loss(self, outputs, targets):
        """
        :param outputs: dict {
            'pred_sub_boxes': torch.Size([3, 64, 4]) = human_pred_boxes,
            'pred_obj_logits': torch.Size([3, 64, 81]) = [batch_size, num_queries, num_classes],
            'pred_obj_boxes': torch.Size([3, 64, 4]) = object_pred_boxes,
            'pred_hoi_logits': torch.Size([3, 64, 600]) = [batch_size, num_queries, classes],
            'action_pred_logits': action_outputs_class[-1],
            'pred_hoi_embedding': torch.Size([batch_size, num_query, 512])
        }
        :param targets: List[dict] {
        }

        :return losses: dict {
            'l1_loss': tensor
            'tri_loss': tensor
        }
        """
        assert 'pred_hoi_embeddings' in outputs
        pred_hoi_embeddings = outputs['pred_hoi_embeddings']
        device = pred_hoi_embeddings.device
        losses = {}

        # label composition
        collate_hoi_embeddings = []
        collate_triplet_labels = []
        for i, t in enumerate(targets):
            # if t['hoi_sentence'][0] == self.special_sentence:
            if len(t['valid_pairs']) == 0:
                collate_query_hoi_embeddings = self.special_sentence_tensor.repeat(1, 100, 1)
                orig = torch.full((100,), 3573)
            else:
                # sol 4
                if len(t['valid_pairs']) == 1:
                    pairs = t['valid_pairs']
                    orig = torch.full((100,), self.pair2tensor_id[pairs[0]])
                    # choice_dict = self.pair_choice_tensor_id[pairs[0]]
                    # verb_choice = [choice_dict['change_verb']] if len(choice_dict['change_verb']) > 0 else []
                    # obj_choice = [choice_dict['change_obj']] if len(choice_dict['change_obj']) > 0 else []
                    # both_choice = [choice_dict['change_both']] if len(choice_dict['change_both']) > 0 else []
                else:
                    pairs = random.choices(t['valid_pairs'], k=10)
                    # print(pairs)
                    # print([self.pair2tensor_id[pair]for pair in pairs])
                    orig = torch.cat([torch.full((10,), self.pair2tensor_id[pair]) for pair in pairs])
                    # print('random')
                verb_choice = [self.pair_choice_tensor_id[pair]['change_verb'] for pair in pairs
                               if len(self.pair_choice_tensor_id[pair]['change_verb']) > 0]
                obj_choice = [self.pair_choice_tensor_id[pair]['change_obj'] for pair in pairs
                              if len(self.pair_choice_tensor_id[pair]['change_obj']) > 0]
                both_choice = [self.pair_choice_tensor_id[pair]['change_both'] for pair in pairs
                               if len(self.pair_choice_tensor_id[pair]['change_both']) > 0]

                verb_mask = torch.rand(100) > (1 - self.p_verb)
                obj_mask = torch.rand(100) > (1 - self.p_obj)
                both_mask = verb_mask * obj_mask
                # print(orig[:20])

                orig = apply_risk_mask(verb_choice, verb_mask, orig)
                orig = apply_risk_mask(obj_choice, obj_mask, orig)
                orig = apply_risk_mask(both_choice, both_mask, orig)

                # # Apply verb
                # if len(verb_choice) > 1:
                #     verb_choice = torch.cat(verb_choice)
                #     verb = verb_choice[torch.randint(len(verb_choice) - 1, size=(100,))]
                #     orig[verb_mask] = verb[verb_mask]
                # elif len(verb_choice) == 1:
                #     verb_choice = torch.cat(verb_choice)
                #     verb = torch.full((100,), verb_choice[0])
                #     orig[verb_mask] = verb[verb_mask]
                #
                # # Apply object
                # if len(obj_choice) > 1:
                #     obj_choice = torch.cat(obj_choice)
                #     obj = obj_choice[torch.randint(len(obj_choice) - 1, size=(100,))]
                #     orig[obj_mask] = obj[obj_mask]
                # elif len(obj_choice) == 1:
                #     obj_choice = torch.cat(obj_choice)
                #     obj = torch.full((100,), obj_choice[0])
                #     orig[obj_mask] = obj[obj_mask]
                #
                # # Apply both
                # if len(both_choice) > 1:
                #     both_choice = torch.cat(both_choice)
                #     both = both_choice[torch.randint(len(both_choice) - 1, size=(100,))]
                #     orig[both_mask] = both[both_mask]
                # elif len(both_choice) == 1:
                #     both_choice = torch.cat(both_choice)
                #     both = torch.full((100,), both_choice[0])
                #     orig[both_mask] = both[both_mask]

                # time 0.005002498626708984
                # print(verb_choice)
                # print(obj_choice)
                # print(both_choice)
                # print(verb_mask[:20])
                # print(obj_mask[:20])
                # print(both_mask[:20])
                # print(orig[:20])
                collate_query_hoi_embeddings = self.text_embeddings[orig].unsqueeze(0)

                # print(collate_query_hoi_embeddings.size())
            collate_triplet_labels.append(orig.unsqueeze(0))
            collate_hoi_embeddings.append(collate_query_hoi_embeddings)
        collate_triplet_labels = torch.cat(collate_triplet_labels, dim=0)  # torch.Size([1, 100])
        collate_hoi_embeddings = torch.cat(collate_hoi_embeddings, dim=0)
        losses['tri_loss'] = batch_hard_triplet_loss(collate_triplet_labels.flatten(),
                                                     collate_hoi_embeddings.flatten(end_dim=1), margin=0.5)
        # losses['tri_loss'] = batch_all_triplet_loss(collate_triplet_labels.flatten(),
        #                                             collate_hoi_embeddings.flatten(end_dim=1), margin=0.5)
        losses['l1_loss'] = self.l1_loss(pred_hoi_embeddings, collate_hoi_embeddings.to(device))

        return losses

    def inference(self, outputs) -> List[dict]:
        assert 'pred_hoi_embeddings' in outputs
        pred_hoi_embeddings = outputs['pred_hoi_embeddings']
        batch_size = pred_hoi_embeddings.size(0)

        pred_hoi_embeddings_norm = F.normalize(pred_hoi_embeddings, dim=-1)  # torch.Size([3, 100, 512)
        sim_matrix = torch.matmul(pred_hoi_embeddings_norm, self.text_embedding_norm.t())  # torch.Size([3, 100, 600])

        # hoi_score sol1
        hoi_score, pred_hoi = torch.max(sim_matrix, dim=-1)  # torch.Size([3, 100])

        # hoi_score sol2: top 100 from 100 * 600
        collate_pred = []
        for i in range(batch_size):
            hoi_scores = sim_matrix[i].cpu().detach().numpy()

            hoi_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))

            hoi_scores = hoi_scores.ravel()
            hoi_labels = hoi_labels.ravel()

            top_k_hoi_scores = top_k(list(hoi_scores), 100)
            top_k_indexes = np.array([np.where(hoi_scores == score)[0][0] for score in top_k_hoi_scores])
            hoi_predictions = [
                {'pred_hoi_label': pred_label, "pred_sentence_score": s, "pred_sentence": self.idx2text[pred_label]}
                for s, pred_label in zip(top_k_hoi_scores, hoi_labels[top_k_indexes])
            ]

            collate_pred.append({
                'hoi_prediction': hoi_predictions
            })

        # return {"pred_hoi": pred_hoi, "collate_pred": collate_pred}
        return collate_pred


if __name__ == '__main__':
    criterion = SentenceCriterion()
    # print(criterion.pair_choice_tensor_id)
    # print(criterion.pair_choice_tensor_id[(27, 40)])
    # print(criterion.pair_choice_tensor_id[(57, 46)])
    # for p, choice_dict in criterion.pair_choice_tensor_id.items():
    #     for k, v in choice_dict.items():
    #         for idx in v:
    #             if idx > ma:
    #                 ma = idx

    batch_size = 1
    target = [{'hoi_sentence': ['a photo of a person lassoing a cow', 'a photo of a person hopping on a bicycle'],
               'valid_pairs': [(27, 40), (57, 46)]
               # 'valid_pairs': [(27, 40)]
               }] * batch_size
    pred = torch.rand(batch_size, 100, 512, requires_grad=True)

    start_time = time.time()

    l1_loss = criterion.batch_l1_con_loss({"pred_hoi_embeddings": pred}, target)
    # l1_loss = criterion.batch_l1_triplet_loss({"pred_hoi_embeddings": pred}, target)


    total_time = time.time() - start_time
    print(f'Training time {total_time}')
    # Training time 0.011998176574707031
    print(l1_loss)

