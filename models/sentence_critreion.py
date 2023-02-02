import os
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

from util.topk import top_k
from .hico_text_label import hico_text_label

# from topk import top_k
# from hico_text_label import hico_text_label


class SentenceCriterion:
    def __init__(self,
                 embedding_file="./checkpoint/hoi_clip_embedding.pth",
                 clip_model="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.device = device

        # pair example: (action_id, object_id)
        self.pair2text = hico_text_label
        self.text2pair = {v: k for k, v in hico_text_label.items()}

        if embedding_file is not None:
            print("load hoi_clip_embedding from pretrained.")
            self.text2tensor = torch.load(embedding_file)
        else:
            text2tensor = {}
            tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            model_proj = CLIPTextModelWithProjection.from_pretrained(clip_model)
            print("hoi_clip_embedding.pth is not given, using clip model to encoding.")
            for i, text in enumerate(self.text2pair.keys()):
                inputs = tokenizer([text], padding=True, return_tensors="pt", max_length=13)
                outputs = model_proj(**inputs)
                text2tensor[text] = F.log_softmax(outputs.text_embeds, dim=1)
            self.text2tensor = text2tensor
            torch.save(text2tensor, "./checkpoint/hoi_clip_embedding_log_space.pth")

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
            print("load special sentence tensor from pretrained.", self.special_sentence_tensor.size())
            self.special_sentence_tensor.to(self.device)

        self.verb2text = {}
        self.text2verb = {}

        self.l1_loss = nn.L1Loss()
        # When using pretrained weight "./checkpoint/hoi_clip_embedding.pth"

        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        # When using pretrained weight "./checkpoint/hoi_clip_embedding_log_space.pth"

        # label composition
        self.p_v = 0.6  # 0.6 to change, 0.4 to not change
        self.k = 10
        self.text_embeddings = torch.cat(list(self.text2tensor.values())).to(self.device)  # torch.Size([600, 512])

        self.text_embedding_norm = F.normalize(self.text_embeddings, dim=-1).to(self.device)
        sim_matrix = torch.mm(self.text_embedding_norm, self.text_embedding_norm.t())
        self.top_k_index = torch.topk(sim_matrix, k=self.k, dim=-1).indices

        self.list_text2tensor = list(self.text2tensor)
        self.text2idx = {text: self.list_text2tensor.index(text) for text in self.text2pair.keys()}
        self.idx2text = {v: k for k, v in self.text2idx.items()}

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
        # :param targets: dict {
        #     'hoi_sentence': List[str],  list of hoi text, ex: ["a photo of a person lassoing a cow", ...].
        #                     len = batch size
        # }
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
                # collate_query_hoi_embeddings = [self.special_sentence_tensor for i in range(100)]
                collate_query_hoi_embeddings = self.special_sentence_tensor.repeat(1, 100, 1).to(self.device)
                # print(collate_query_hoi_embeddings.size())
            else:
                embedding_idx = self.text2idx[text]
                top_k_indices = self.top_k_index[embedding_idx]
                # tensor([101, 104, 103,  98,  96, 105, 106,  97, 100, 102]) first one is original, e.g. 101

                mask = torch.rand(100) > self.p_v
                rand_idxs = torch.randint(1, len(top_k_indices), size=(1, 100))[0]
                rand_idxs[mask] = 0  # keep original embedding
                rand_similar_label = top_k_indices[rand_idxs]

                # problematic
                # collate_query_hoi_embeddings = [self.text_embeddings[idx].view(1, -1)
                #                                 for idx in rand_similar_label]
                collate_query_hoi_embeddings = self.text_embeddings[rand_similar_label].unsqueeze(0)

            # collate_query_hoi_embeddings = torch.stack(collate_query_hoi_embeddings, dim=1)
            # torch.Size([1, num_query, 512]) = torch.Size([1, 100, 512])

            # if torch.rand(1) < self.p_v:
            #     # random pick embeddings from top k neighbor
            #     embedding_idx = self.text2idx[text]
            #     # embedding_idx = self.list_text2tensor.index(text)
            #     top_k_indices = self.top_k_index[embedding_idx]
            #     rand_idx = torch.randint(1, self.k, (1, 1))[0, 0]
            #     rand_embedding_idx = top_k_indices[rand_idx]
            #
            #     # print(f"embedding_idx: {embedding_idx}")
            #     # print(f"top_k_indices: {top_k_indices}")
            #     # print(f"rand_idx {rand_idx} rand_embedding_idx {rand_embedding_idx}")
            #
            #     collate_hoi_embeddings.append(self.text_embeddings[rand_embedding_idx].view(1, -1))
            # else:
            #     collate_hoi_embeddings.append(self.text2tensor[text])
            collate_hoi_embeddings.append(collate_query_hoi_embeddings)

        collate_hoi_embeddings = torch.cat(collate_hoi_embeddings, dim=0)
        losses['l1_loss'] = self.l1_loss(pred_hoi_embeddings, collate_hoi_embeddings.to(device))

        return losses

    def batch_kl_loss(self, outputs, targets):
        """
        :param outputs: dict {
            'pred_sub_boxes': torch.Size([3, 64, 4]) = human_pred_boxes,
            'pred_obj_logits': torch.Size([3, 64, 81]) = [batch_size, num_queries, num_classes],
            'pred_obj_boxes': torch.Size([3, 64, 4]) = object_pred_boxes,
            'pred_hoi_logits': torch.Size([3, 64, 600]) = [batch_size, num_queries, classes],
            'action_pred_logits': action_outputs_class[-1],
            # TODO: modify for torch.Size([batch_size, **num_query**, 512])
            'pred_hoi_embedding': torch.Size([batch_size, 512])
        }
        :param targets: dict {
            'hoi_sentence': list of hoi text label,   ex: ["a photo of a person lassoing a cow", ]
        }

        :return losses: dict {

        }
        """
        assert 'pred_hoi_embeddings' in outputs
        assert 'hoi_sentence' in targets
        pred_hoi_embeddings = outputs['pred_hoi_embeddings']

        # KL loss
        losses = {}
        collate_hoi_text = [self.text2tensor[text] for text in targets['hoi_sentence']]
        collate_hoi_embedding = torch.cat(collate_hoi_text, dim=0)

        losses['kl_div'] = self.kl_loss(F.log_softmax(pred_hoi_embeddings, dim=1), collate_hoi_embedding)

        return losses

    def inference(self, outputs) -> List[dict]:
        """
        :param outputs: dict {
                'pred_sub_boxes': torch.Size([3, 64, 4]) = human_pred_boxes,
                'pred_obj_logits': torch.Size([3, 64, 81]) = [batch_size, num_queries, num_classes],
                'pred_obj_boxes': torch.Size([3, 64, 4]) = object_pred_boxes,
                'pred_hoi_logits': torch.Size([3, 64, 600]) = [batch_size, num_queries, classes],
                'action_pred_logits': action_outputs_class[-1],
                'pred_hoi_embedding': torch.Size([batch_size, num_query, 512])
            }
        : return: dict {
                "pred_hoi": torch.tensor, # torch.Size([batch_size, num_query])
            }
        """
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

    # target = criterion.text2tensor['a photo of a person lassoing a cow']
    # pred = F.softmax(torch.rand(10, 512, requires_grad=True), dim=1)
    # target = F.softmax(torch.rand(10, 512), dim=1)
    # print(pred)
    # print(target)

    target = [{'hoi_sentence': ['a photo of a person lassoing a cow']}] * 3
    pred = torch.rand(3, 100, 512, requires_grad=True)
    # kl_loss = criterion.batch_kl_loss({"pred_hoi_embeddings": pred}, {"hoi_sentence": target})
    l1_loss = criterion.batch_l1_loss({"pred_hoi_embeddings": pred}, target)
    print(l1_loss)
    # criterion.inference({"pred_hoi_embeddings": pred})

    # a = torch.log_softmax(torch.randn(64, 81), dim=1)
    # b = torch.softmax(torch.randn(64, 81), dim=1)
    # criterion = nn.KLDivLoss()
    # loss = criterion(a, b)
    # print(loss)
