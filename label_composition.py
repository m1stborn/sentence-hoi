import torch
import torch.nn.functional as F

from dataclasses import dataclass
from models.sentence_critreion import SentenceCriterion


@dataclass
class LabelCompose:
    p_v: float = 0.8  # 0.8 change verb, 0.2 not change
    p_o: float = 0.2
    epoch: int = 200
    t_sim: float = 0.7
    lr_decay: int = 150

    # not sure
    top_k: int = 3


if __name__ == '__main__':
    se = SentenceCriterion()
    text_embedding = torch.cat(list(se.text2tensor.values()))
    text_embedding_norm = F.normalize(text_embedding, dim=-1)
    sim_matrix = torch.mm(text_embedding_norm, text_embedding_norm.t())

    # top_k = torch.topk(sim_matrix, k=5, dim=-1).indices
    # print(torch.topk(sim_matrix, k=5, dim=-1).indices[0])
    #
    # print(text_embedding[top_k[0]].size())
    # print(((torch.topk(sim_matrix, k=10, dim=-1).values > 0.7) == 0).nonzero().size())  # number < 0.7: 1
    # print(list(se.text2tensor.keys())[:20])

    # rand_idx = torch.randint(1, 10, (1, 1))[0, 0]
    # print(text_embedding[rand_idx], rand_idx)

    target = ['a photo of a person lassoing a cow'] * 3
    pred = torch.rand(3, 512, requires_grad=True)

    import time
    start_time = time.time()
    kl_loss = se.batch_l1_loss({"pred_hoi_embeddings": pred}, {"hoi_sentence": target})
    total_time = time.time() - start_time
    print('Training time {}'.format(total_time))
