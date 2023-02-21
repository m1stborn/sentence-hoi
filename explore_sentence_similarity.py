import json
from itertools import product

import torch
import torch.nn.functional as F

from models.hico_text_label import hico_text_label
from label_composition import ing_form


def main():
    ckpt = torch.load("checkpoint/synth_hoi_clip_embedding_ckpt.pth")
    text_embeddings = ckpt["text_tensor"]
    text2idx = ckpt["sentence2tensor_id"]
    tensor_id2sentence = {v: k for k, v in text2idx.items()}
    print(len(tensor_id2sentence))
    # pair2tensor_id = ckpt["pair2tensor_id"]
    # real2synth_tensor_id = ckpt['real2synth_tensor_id']

    # pair2tensor_id_narrow = {k: v for k, v in pair2tensor_id.items() if k in hico_text_label}
    # print(f"Len of pair2tensor_id_narrow {len(pair2tensor_id_narrow)}")
    # idx2text = {v: k for k, v in text2idx.items()}
    # real2synth_tensor_id = ckpt['real2synth_tensor_id']
    # real2synth_tensor_id = {k: torch.tensor(v) for k, v in real2synth_tensor_id.items()}

    with open("data/similar_word/obj2id.json", encoding='utf-8') as file:
        obj2id = json.load(file)
    with open("data/similar_word/verb2id.json", encoding='utf-8') as file:
        verb2id = json.load(file)

    id2verb = {v: k for k, v in verb2id.items()}
    id2obj = {v: k for k, v in obj2id.items()}

    # Load verb choice
    with open("data/similar_word/similar_verb_google.json", encoding='utf-8') as file:
        sim_verb = json.load(file)

    # Load obj choice
    with open("data/similar_word/similar_obj_google.json", encoding='utf-8') as file:
        sim_obj = json.load(file)

    verb_idx = len(id2verb.keys())
    obj_idx = len(id2obj.keys())
    for k, v in sim_verb.items():
        if k == "no_interaction":
            continue
        for sim, score in v.items():
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

    text_embedding_norm = F.normalize(text_embeddings, dim=-1)
    sim_matrix = torch.mm(text_embedding_norm, text_embedding_norm.t())

    print(f"Similarity Matrix Size: {sim_matrix.size()}")

    pair_choice = {}

    synth_pair_pool = []
    real2synth = {}
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
            pair_choice[(verb_id, obj_id)] = {
                "change_verb": [(verb2id[v], obj2id[o]) for v, o in synth if o == verb_id],
                "change_obj": [(verb2id[v], obj2id[o]) for v, o in synth if v == obj_id],
                "change_both": [(verb2id[v], obj2id[o]) for v, o in synth if o != obj_id and v != verb_id],
            }
        real2synth[(verb_id, obj_id)] = [(verb2id[v], obj2id[o]) for v, o in synth]

    pair2tensor_id = {}
    pair2sentence = {}
    idx = 0
    for i, (verb, obj) in enumerate(synth_pair_pool):
        pair = (verb2id[verb], obj2id[obj])
        if pair in pair2tensor_id:
            continue
        pair2tensor_id[pair] = idx

        if pair in hico_text_label:
            text = hico_text_label[pair]
            pair2sentence[pair] = text
        else:
            v_ing = ing_form(verb) if '_' not in verb else ing_form(verb.split('_')[0]) + ' ' + verb.split('_')[1]
            v_ing = v_ing.replace('noing interaction', 'and')
            text = f"a photo of a person {v_ing} {'an' if obj[0] in 'aeiou' else 'a'}" \
                   f" {obj.replace('_', ' ')}."
            pair2sentence[pair] = text
            idx += 1

    pair_choice_tensor_id = {}
    for pair, choice_dict in pair_choice.items():
        pair_choice_tensor_id[pair] = {
            k: torch.tensor([pair2tensor_id[sub_pair] for sub_pair in v]) for k, v in choice_dict.items()
        }

    print(pair_choice_tensor_id)
    ckpt['pair_choice_tensor_id'] = pair_choice_tensor_id
    torch.save(ckpt, "./checkpoint/pari_choice_clip_embedding_ckpt.pth")

    # Sanity check
    ckp1 = torch.load("./checkpoint/synth_hoi_clip_embedding_ckpt.pth")
    ckp2 = torch.load("./checkpoint/pari_choice_clip_embedding_ckpt.pth")

    assert torch.equal(ckp1['text_tensor'], ckp2['text_tensor'])

    # print(len(pair2tensor_id))
    # similar_sentence = {
    #     pair2sentence[k]: {
    #         pair2sentence[synth]: sim_matrix[pair2tensor_id[k], pair2tensor_id[synth]].item()
    #         for synth in v
    #     }
    #     for k, v in real2synth.items()
    # }

    # with open("data/similar_sentence.json", "w", encoding="utf-8") as file:
    #     json.dump(similar_sentence, file, ensure_ascii=False, indent=4)

    return


if __name__ == "__main__":
    main()
