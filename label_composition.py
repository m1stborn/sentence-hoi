import json
import re
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import pandas as pd

from models.sentence_critreion import SentenceCriterion
from datasets.hico_text_label import hico_text_label
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


def main():
    # config
    clip_model = "openai/clip-vit-base-patch32"
    special_sentence = "A photo of a person."
    template = "A photo of a person verb obj."

    with open("./data/similar_word/obj2id.json", encoding='utf-8') as file:
        obj2id = json.load(file)
    with open("./data/similar_word/verb2id.json", encoding='utf-8') as file:
        verb2id = json.load(file)

    id2verb = {v: k for k, v in verb2id.items()}
    id2obj = {v: k for k, v in obj2id.items()}

    # Load verb choice
    with open("./data/similar_word/similar_verb_google.json", encoding='utf-8') as file:
        sim_verb = json.load(file)

    # Load obj choice
    with open("./data/similar_word/similar_obj_google.json", encoding='utf-8') as file:
        sim_obj = json.load(file)

    verb_idx = len(id2verb.keys())
    obj_idx = len(id2obj.keys())
    for k, v in sim_verb.items():
        if k == "no_interaction":
            continue
        for sim, score in v.items():
            # if sim == "NO":

            if score > 0.7 and sim not in verb2id:
                verb2id[sim] = verb_idx
                verb_idx += 1

    for k, v in sim_obj.items():
        for sim, score in v.items():
            if score > 0.7 and sim not in obj2id:
                obj2id[sim] = obj_idx
                obj_idx += 1

    id2verb = {v: k for k, v in verb2id.items()}
    id2obj = {v: k for k, v in obj2id.items()}
    sim_verb['no_interaction'] = {}

    # with open("data/similar_word/extend_verb2id.json", "w", encoding="utf-8") as file:
    #     json.dump(verb2id, file, ensure_ascii=False, indent=4)
    #
    # with open("data/similar_word/extend_obj2id.json", "w", encoding="utf-8") as file:
    #     json.dump(obj2id, file, ensure_ascii=False, indent=4)
    #
    # with open("./data/similar_word/similar_verb_idx.json", "w", encoding="utf-8") as file:
    #     json.dump({int(verb2id[k]): [verb2id[k]] + [verb2id[sim] for sim, score in v.items() if score > 0.7]
    #                for k, v in sim_verb.items()},
    #               file, ensure_ascii=False, indent=4)
    #
    # with open("./data/similar_word/similar_obj_idx.json", "w", encoding="utf-8") as file:
    #     json.dump({int(obj2id[k]): [obj2id[k]] + [obj2id[sim] for sim, score in v.items() if score > 0.7]
    #                for k, v in sim_obj.items()},
    #               file, ensure_ascii=False, indent=4)

    # pooling
    synth_pair_pool = []
    real2synth = {}
    real2text = {}
    for i, ((verb_id, obj_id), text) in enumerate(hico_text_label.items()):
        verb = id2verb[verb_id]
        obj = id2obj[obj_id]
        verb_choice = [k for k, v in sim_verb[verb].items() if v > 0.7]
        verb_choice.append(verb)  # add back self
        obj_choice = [k for k, v in sim_obj[obj].items() if v > 0.7]
        obj_choice.append(obj)
        synth = list(product(verb_choice, obj_choice))
        if len(verb_choice) > 0 and len(obj_choice) > 0:
            synth_pair_pool.extend(synth)
        real2synth[(verb_id, obj_id)] = [(verb2id[v], obj2id[o]) for v, o in synth]

    print(len(synth_pair_pool))
    df = pd.DataFrame.from_dict({"id": [i for i in range(len(synth_pair_pool))],
                                 "synth_pair": synth_pair_pool})
    df.to_csv("./data/similar_word/synth_pair.csv", index=False)

    tokenizer = CLIPTokenizer.from_pretrained(clip_model)
    model_proj = CLIPTextModelWithProjection.from_pretrained(clip_model)

    text_tensor = []
    pair2tensor_id = {}
    idx = 0
    for i, (verb, obj) in enumerate(synth_pair_pool):
        pair = (verb2id[verb], obj2id[obj])
        if pair in pair2tensor_id:
            continue
        pair2tensor_id[pair] = idx

        if pair in hico_text_label:
            print(pair)
            text = hico_text_label[pair]
        else:
            v_ing = ing_form(verb) if '_' not in verb else ing_form(verb.split('_')[0]) + ' ' + verb.split('_')[1]
            v_ing = v_ing.replace('noing interaction', 'and')
            text = f"a photo of a person {v_ing} {'an' if obj[0] in 'aeiou' else 'a'}" \
                   f" {obj.replace('_', ' ')}."
        # verb = verb.replace('no_interaction', 'and')
        # text = f"A photo of a person {verb.replace('_', ' ')} {obj.replace('_', ' ')}."
        # pair2tensor_id[pair] = idx

        # inputs = tokenizer([text], padding=True, return_tensors="pt", max_length=16)
        # outputs = model_proj(**inputs)
        # idx += 1
        # text_tensor.append(outputs.text_embeds)

    # text_tensor = torch.cat(text_tensor)
    # print(text_tensor.size())

    ckpt = torch.load("./checkpoint/synth_hoi_clip_embedding_ckpt.pth")
    text_tensor = ckpt["text_tensor"]
    pair2tensor_id = ckpt["pair2tensor_id"]
    tensor_id2pair = {v: k for k, v in pair2tensor_id.items()}
    # print(real2synth)

    real2synth_tensor_id = {pair2tensor_id[k]: [pair2tensor_id[k]] + [pair2tensor_id[p] for p in synth_pairs[:-1]]
                            for k, synth_pairs in real2synth.items()}
    print(real2synth_tensor_id)
    # real2sentence = {k: v for k, v in real2synth_tensor_id}

    torch.save({
        "text_tensor": text_tensor,
        "pair2tensor_id": pair2tensor_id,
        'sentence2tensor_id': {text: pair2tensor_id[p] for p, text in hico_text_label.items()},
        "real2synth_tensor_id": real2synth_tensor_id,  # include self
    }, "./checkpoint/synth_hoi_gpt-1_embedding_ckpt.pth")

    # with open("data/similar_word/pair2id.txt", "w", encoding="utf-8") as f:
    #     for k, v in pair2tensor_id.items():
    #         print(k, v, file=f)

    # Load embedding
    # ckpt = torch.load("./checkpoint/synth_hoi_clip_embedding_ckpt.pth")
    # text_embeddings = ckpt["text_tensor"]
    # pair2tensor_id = ckpt["pair2tensor_id"]
    # tensor_id2pair = {v: k for k, v in pair2tensor_id.items()}
    #
    # text_embedding_norm = F.normalize(text_embeddings, dim=-1)
    # sim_matrix = torch.mm(text_embedding_norm, text_embedding_norm.t())
    # top_k = torch.topk(sim_matrix, k=30, dim=-1)
    # top_k_score = top_k.values
    # top_k_index = top_k.indices
    # print(top_k_index)
    # sim_sentence = {}
    #
    # vowel = ['a', 'e', 'i', 'o', 'u']
    # tensor_id2sentence = {}
    # for i, ((v_id, obj_id), idx) in enumerate(pair2tensor_id.items()):
    #     verb, obj = id2verb[v_id], id2obj[obj_id]
    #     v_ing = ing_form(verb) if '_' not in verb else ing_form(verb.split('_')[0]) + ' ' + verb.split('_')[1]
    #     v_ing = v_ing.replace('noing interaction', 'and')
    #     text = f"a photo of a person {v_ing} {'an' if obj[0] in vowel else 'a'}" \
    #            f" {obj.replace('_', ' ')}"
    #     tensor_id2sentence[idx] = text
    #
    # for i, (query, text) in enumerate(hico_text_label.items()):
    #     k = pair2tensor_id[query]
    #     v = tensor_id2sentence[k]
    #     sim_sentence[v] = {tensor_id2sentence[i.item()]: s.item() for i, s in zip(top_k_index[k], top_k_score[k])}
    #
    # with open("data/similar_word/sentence.json", "w", encoding="utf-8") as file:
    #     json.dump(sim_sentence, file, ensure_ascii=False, indent=4)


def ing_form(s):
    li = []
    # print(ingFrom(input('Enter Verb:')))
    for x in s:
        li.append(x)
    if li[-4:] == "ing":
        # print(li)
        return "".join(li)
    if li[len(li) - 1] == 'e' and li[len(li) - 2] != 'i':
        del li[len(li) - 1]
        li.append("ing")
    elif li[len(li) - 1] == 'e' and li[len(li) - 2] == 'i':
        del li[len(li) - 1]
        del li[len(li) - 1]
        li.append("ying")
        """To Check"""
    elif li[len(li) - 2] in 'aeiou' and li[len(li) - 1] not in 'aeiou':
        temp = li[len(li) - 1]
        del li[len(li) - 1]
        li.append(temp)
        if li[len(li) - 3] not in 'xaeiou':
            li.append(temp)
        li.append("ing")
    elif li[len(li) - 1] in 'aeiouy':
        li.append("ing")
    else:
        li.append("ing")
    return "".join(li)


# li=[]
# print(ingFrom(input('Enter Verb:')))


if __name__ == '__main__':
    main()
    # synth_ckpt = torch.load("./checkpoint/synth_hoi_clip_embedding_ckpt.pth")
    # print(len(synth_ckpt["pair2tensor_id"].keys()))
