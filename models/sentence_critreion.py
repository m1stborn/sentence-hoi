import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

from .hico_text_label import hico_text_label


class SentenceCriterion:
    def __init__(self,
                 embedding_file="./checkpoint/hoi_clip_embedding.pth",
                 clip_model="openai/clip-vit-base-patch32"):
        super().__init__()

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

        self.verb2text = {}
        self.text2verb = {}

        self.l1_loss = nn.L1Loss()
        # When using pretrained weight "./checkpoint/hoi_clip_embedding.pth"

        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        # When using pretrained weight "./checkpoint/hoi_clip_embedding_log_space.pth"

        # label composition
        self.p_v = 0.6  # 0.6 to change, 0.4 to not change
        self.k = 10

        self.text_embeddings = torch.cat(list(self.text2tensor.values()))  # torch.Size([600, 512])

        text_embedding_norm = F.normalize(self.text_embeddings, dim=-1)
        sim_matrix = torch.mm(text_embedding_norm, text_embedding_norm.t())
        self.top_k_index = torch.topk(sim_matrix, k=self.k, dim=-1).indices

        self.list_text2tensor = list(self.text2tensor)
        self.text2idx = {text: self.list_text2tensor.index(text) for text in self.text2pair.keys()}

    def batch_l1_loss(self, outputs, targets):
        """
        :param outputs: dict {
            'pred_sub_boxes': torch.Size([3, 64, 4]) = human_pred_boxes,
            'pred_obj_logits': torch.Size([3, 64, 81]) = [batch_size, num_queries, num_classes],
            'pred_obj_boxes': torch.Size([3, 64, 4]) = object_pred_boxes,
            'pred_hoi_logits': torch.Size([3, 64, 600]) = [batch_size, num_queries, classes],
            'action_pred_logits': action_outputs_class[-1],
            'pred_hoi_embedding': torch.Size([batch_size, 512])
        }
        :param targets: dict {
            'hoi_sentence': List[str],  list of hoi text, ex: ["a photo of a person lassoing a cow", ...]
        }

        :return losses: dict {

        }
        """
        assert 'pred_hoi_embeddings' in outputs
        assert 'hoi_sentence' in targets
        pred_hoi_embeddings = outputs['pred_hoi_embeddings']

        losses = {}

        # collate_hoi_embeddings = [self.text2tensor[text] for text in targets['hoi_sentence']]
        # label composition
        collate_hoi_embeddings = []
        for i, text in enumerate(targets['hoi_sentence']):
            if torch.rand(1) < self.p_v:
                # random pick embeddings from top k neighbor
                embedding_idx = self.text2idx[text]
                # embedding_idx = self.list_text2tensor.index(text)
                top_k_indices = self.top_k_index[embedding_idx]
                rand_idx = torch.randint(1, self.k, (1, 1))[0, 0]
                rand_embedding_idx = top_k_indices[rand_idx]

                # print(f"embedding_idx: {embedding_idx}")
                # print(f"top_k_indices: {top_k_indices}")
                # print(f"rand_idx {rand_idx} rand_embedding_idx {rand_embedding_idx}")

                collate_hoi_embeddings.append(self.text_embeddings[rand_embedding_idx].view(1, -1))
            else:
                collate_hoi_embeddings.append(self.text2tensor[text])

        collate_hoi_embeddings = torch.cat(collate_hoi_embeddings, dim=0)

        losses['l1_loss'] = self.l1_loss(pred_hoi_embeddings, collate_hoi_embeddings)

        # print(F.l1_loss(pred_hoi_embeddings, collate_hoi_embeddings), losses['l1_loss'])

        return losses

    def batch_kl_loss(self, outputs, targets):
        """
        :param outputs: dict {
            'pred_sub_boxes': torch.Size([3, 64, 4]) = human_pred_boxes,
            'pred_obj_logits': torch.Size([3, 64, 81]) = [batch_size, num_queries, num_classes],
            'pred_obj_boxes': torch.Size([3, 64, 4]) = object_pred_boxes,
            'pred_hoi_logits': torch.Size([3, 64, 600]) = [batch_size, num_queries, classes],
            'action_pred_logits': action_outputs_class[-1],
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

    # TODO: implement look up table
    def inference(self):
        return

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
        :param targets: dict {
            'hoi_sentence': List[str],  list of hoi text, ex: ["a photo of a person lassoing a cow", ...]
        }

        :return losses: dict {

        }
        """
        assert 'pred_hoi_embeddings' in outputs
        assert 'hoi_sentence' in targets

        losses = {}

        return losses


if __name__ == '__main__':
    criterion = SentenceCriterion()

    # target = criterion.text2tensor['a photo of a person lassoing a cow']
    # pred = F.softmax(torch.rand(10, 512, requires_grad=True), dim=1)
    # target = F.softmax(torch.rand(10, 512), dim=1)
    # print(pred)
    # print(target)

    target = ['a photo of a person lassoing a cow'] * 3
    pred = torch.rand(3, 512, requires_grad=True)
    kl_loss = criterion.batch_kl_loss({"pred_hoi_embeddings": pred}, {"hoi_sentence": target})
    l1_loss = criterion.batch_l1_loss({"pred_hoi_embeddings": pred}, {"hoi_sentence": target})

    # a = torch.log_softmax(torch.randn(64, 81), dim=1)
    # b = torch.softmax(torch.randn(64, 81), dim=1)
    # criterion = nn.KLDivLoss()
    # loss = criterion(a, b)
    # print(loss)
