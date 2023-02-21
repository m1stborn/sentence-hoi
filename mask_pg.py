import math
import time

import torch
import torch.nn.functional as F


def main():

    # mask = torch.rand(10) > 0.5
    # a = torch.rand(10)
    # b = torch.rand(10)
    # c = a * mask + b * ~mask
    # print(mask)
    # print(a)
    # print(b)
    # print(c)
    # start_time = time.time()
    #
    # sim_idx = [3, 4, 7, 10, 2, 2, 2, 3, 4, 3, 4, 7, 10, 2, 2, 2, 3, 4]
    # hoi_len = len(sim_idx)
    # split_len = math.ceil(100/hoi_len)
    # print(f"split_len: {split_len}")
    #
    # embed_idx = [torch.rand(l) for l in sim_idx]
    # collate_a = []
    # collate_b = []
    # for i, (idx_len, embed) in enumerate(zip(sim_idx, embed_idx)):
    #     pad_a = torch.tile(embed, (math.ceil(split_len/idx_len),))[:split_len]
    #     pad_b = torch.full((split_len, ), i)
    #     collate_a.append(pad_a)
    #     collate_b.append(pad_b)
    # collate_a = torch.cat(collate_a)[:100]
    # collate_b = torch.cat(collate_b)[:100]
    # print(collate_b)
    #
    # mask = torch.rand(100) > 0.5
    # c = collate_a * mask + collate_b * ~mask
    #
    # total_time = time.time() - start_time
    # print(f'Training time {total_time}')
    # ckpt = torch.load("checkpoint/p_202302190233/checkpoint_best.pth")
    # print(ckpt['args'])
    # ckpt = torch.load("./checkpoint/synth_hoi_clip_embedding_ckpt.pth")
    # text_embeddings = ckpt["text_tensor"]
    # text2idx = ckpt["sentence2tensor_id"]
    # pair2idx = ckpt["pair2tensor_id"]
    # print(pair2idx)
    # idx2text = {v: k for k, v in text2idx.items()}
    # real2synth_tensor_id = ckpt['real2synth_tensor_id']
    # real2synth_tensor_id = {k: torch.tensor(v) for k, v in real2synth_tensor_id.items()}
    start_time = time.time()

    orig = torch.randint(0, 9, size=(10, ))
    verb = torch.randint(10, 19, size=(10, ))
    obj = torch.randint(20, 29, size=(10, ))
    both = torch.randint(30, 39, size=(10, ))

    verb_mask = torch.rand(10) > 0.2
    obj_mask = torch.rand(10) > 0.8
    both_mask = verb_mask*obj_mask

    orig[verb_mask] = verb[verb_mask]
    orig[obj_mask] = obj[obj_mask]
    orig[both_mask] = both[both_mask]

    # print(verb_mask)
    # print(obj_mask)
    # print(orig)

    total_time = time.time() - start_time
    print(f'Total time {total_time}')
    # print(synth[mask, :].size())
    return


if __name__ == "__main__":
    main()
