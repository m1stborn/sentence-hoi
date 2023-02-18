import json
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

# from util.topk import top_k
# from .hico_text_label import hico_text_label

# from topk import top_k
# from hico_text_label import hico_text_label


class SentenceCriterion:
    def __init__(self,
                 embedding_file="./checkpoint/synth_hoi_clip_embedding_ckpt.pth",
                 clip_model="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.device = device

        # pair example: (action_id, object_id)
        # self.pair2text = hico_text_label
        # self.text2pair = {v: k for k, v in hico_text_label.items()}

        if embedding_file is not None:
            print("load synth_hoi_clip_embedding from pretrained.")
            ckpt = torch.load(embedding_file)
            self.text_embeddings = ckpt["text_tensor"]
            self.pair2tensor_id = ckpt["pair2tensor_id"]

        # TODO: make path to argument
        # special sentence
        self.special_sentence = "A photo of a person."
        if not os.path.exists("./checkpoint/special_sentence_tensor.pth"):
            tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            model_proj = CLIPTextModelWithProjection.from_pretrained(clip_model)
            inputs = tokenizer([self.special_sentence], padding=True, return_tensors="pt", max_length=13)
            outputs = model_proj(**inputs)
            # print(outputs.text_embeds, outputs.text_embeds.size())
            torch.save({"A photo of a person.": outputs.text_embeds}, "./checkpoint/special_sentence_tensor.pth")
            self.special_sentence_tensor = outputs.text_embeds
        else:
            self.special_sentence_tensor = torch.load("./checkpoint/special_sentence_tensor.pth")[self.special_sentence]
            print("Load special sentence tensor from pretrained.", self.special_sentence_tensor.size())
            # self.special_sentence_tensor.to(self.device)

        self.l1_loss = nn.L1Loss()
        # When using pretrained weight "./checkpoint/hoi_clip_embedding.pth"

        # label composition
        self.p_v = 0.8  # 0.6 to change, 0.4 to not change
        self.p_o = 0.2
        # self.text_embeddings = torch.cat(list(self.text2tensor.values())).to(self.device)  # torch.Size([600, 512])
        with open("./data/similar_word/similar_verb_idx.json", encoding='utf-8') as file:
            top_k_similar_verb = json.load(file)
        with open("./data/similar_word/similar_obj_idx.json", encoding='utf-8') as file:
            top_k_similar_obj = json.load(file)

        with open("./data/similar_word/similar_obj_idx.json", encoding='utf-8') as file:
            top_k_similar_obj = json.load(file)

        self.top_k_similar_verb = {k: torch.tensor(v) for k, v in top_k_similar_verb.items()}
        self.top_k_similar_obj = {k: torch.tensor(v) for k, v in top_k_similar_obj.items()}

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
            'hoi_pair: ["a photo of a person lassoing a cow", ...]
        }

        :return losses: dict {
            'l1_loss': tensor
        }
        """
        assert 'pred_hoi_embeddings' in outputs

        pred_hoi_embeddings = outputs['pred_hoi_embeddings']
        device = pred_hoi_embeddings.device
        losses = {}

        # label composition
        collate_hoi_embeddings = []
        for i, t in enumerate(targets):
            # TODO: utilize rest of the hoi_pair
            if len(t['hoi_pairs']) == 0:
                # TODO: using obj embedding
                collate_query_hoi_embeddings = self.special_sentence_tensor.repeat(1, 100, 1)
            else:
                pair = t['hoi_pairs'][0]  # TODO: utilize rest of the pair
                top_k_verb = self.top_k_similar_verb[str(pair[0])]
                if len(top_k_verb) <= 1:
                    verb_idxs = torch.full((1, 100), pair[0])[0]
                else:
                    mask_verb = torch.rand(100) > self.p_v
                    rand_verb_idxs = torch.randint(1, len(top_k_verb), size=(1, 100))[0]
                    rand_verb_idxs[mask_verb] = 0
                    verb_idxs = top_k_verb[rand_verb_idxs]

                top_k_obj = self.top_k_similar_obj[str(pair[1])]
                if len(top_k_obj) <= 1:
                    obj_idxs = torch.full((1, 100), pair[1])[0]
                else:
                    mask_obj = torch.rand(100) > self.p_o
                    rand_obj_idxs = torch.randint(1, len(top_k_obj), size=(1, 100))[0]
                    rand_obj_idxs[mask_obj] = 0
                    obj_idxs = top_k_obj[rand_obj_idxs]
                pair_query = list(zip(verb_idxs.tolist(), obj_idxs.tolist()))
                # print(pair, list(zip(rand_verb_idxs.tolist(), rand_obj_idxs.tolist())))
                idx = [self.pair2tensor_id[p] for p in pair_query]
                collate_query_hoi_embeddings = self.text_embeddings[idx].unsqueeze(0)

            collate_hoi_embeddings.append(collate_query_hoi_embeddings)

        collate_hoi_embeddings = torch.cat(collate_hoi_embeddings, dim=0)
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

    # TODO
    def triplet_loss(self, outputs, targets):
        """
        :param outputs: dict {
            'pred_sub_boxes': torch.Size([3, 64, 4]) = human_pred_boxes,
            'pred_obj_logits': torch.Size([3, 64, 81]) = [batch_size, num_queries, num_classes],
            'pred_obj_boxes': torch.Size([3, 64, 4]) = object_pred_boxes,
            'pred_hoi_logits': torch.Size([3, 64, 600]) = [batch_size, num_queries, classes],
            'action_pred_logits': action_outputs_class[-1],
            'pred_hoi_embedding': torch.Size([batch_size, 512])
        }
        :param targets: List[dict] {
            'hoi_sentence: ["a photo of a person lassoing a cow", ...]
        }
        # :param targets: dict {
        #     'hoi_sentence': List[str],  list of hoi text, ex: ["a photo of a person lassoing a cow", ...]
        # }
        :return losses: dict {

        }
        """
        assert 'pred_hoi_embeddings' in outputs
        # assert 'hoi_sentence' in targets

        losses = {}

        return losses


if __name__ == '__main__':
    criterion = SentenceCriterion()

    target = [{'hoi_pair': [(4, 4)]}] * 16
    pred = torch.rand(16, 100, 512, requires_grad=True)

    start_time = time.time()

    l1_loss = criterion.batch_l1_loss({"pred_hoi_embeddings": pred}, target)

    total_time = time.time() - start_time
    print(f'Training time {total_time}')
    # Training time 0.011998176574707031
    print(l1_loss)

