import json
from itertools import product

import gensim.downloader
import torch
import torch.nn.functional as F

from label_composition import ing_form
from models.hico_text_label import hico_text_label


def main():
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
        real2synth[(verb_id, obj_id)] = [(verb2id[v], obj2id[o]) for v, o in synth]

    glove_vectors = gensim.downloader.load('word2vec-google-news-300')
    person_tensor = torch.tensor(glove_vectors['person'])

    synth_tensor = []
    pair2tensor_id = {}
    idx = 0
    pair2sentence = {}
    word_tensor = {}

    def get_tensor(word):
        if word in word_tensor:
            return word_tensor[word]
        else:
            try:
                vec = torch.tensor(glove_vectors[word])
            except KeyError:
                vec = torch.tensor(glove_vectors[word.split("_")[0]])
            word_tensor[word] = vec
            return vec

    for i, (verb, obj) in enumerate(synth_pair_pool):
        pair = (verb2id[verb], obj2id[obj])
        print(pair)
        if pair in pair2tensor_id:
            continue
        pair2tensor_id[pair] = idx
        pair2sentence[pair] = verb + ' ' + obj

        verb_tensor = get_tensor(verb)
        obj_tensor = get_tensor(obj)
        synth_tensor.append(torch.cat([person_tensor, verb_tensor, obj_tensor]))

        idx += 1

    synth_tensor = torch.stack(synth_tensor)
    print(synth_tensor.size())
    torch.save({
        "text_tensor": synth_tensor,
    }, "./checkpoint/synth_hoi_word2vec_embedding_ckpt.pth")

    text_embedding_norm = F.normalize(synth_tensor, dim=-1)
    sim_matrix = torch.mm(text_embedding_norm, text_embedding_norm.t())

    print(f"Similarity Matrix Size: {sim_matrix.size()}")

    real2synth_tensor_id = {pair2tensor_id[k]: [pair2tensor_id[k]] + [pair2tensor_id[p] for p in synth_pairs]
                            for k, synth_pairs in real2synth.items()}

    torch.save({
        "text_tensor": synth_tensor,
        "pair2tensor_id": pair2tensor_id,
        'sentence2tensor_id': {text: pair2tensor_id[p] for p, text in hico_text_label.items()},
        "real2synth_tensor_id": real2synth_tensor_id,  # include self
    }, "./checkpoint/synth_hoi_word2vec_embedding_ckpt.pth")

    similar_sentence = {
        pair2sentence[k]: {
            pair2sentence[synth]: sim_matrix[pair2tensor_id[k], pair2tensor_id[synth]].item()
            for synth in v
        }
        for k, v in real2synth.items()
    }

    with open("./data/similar_word.json", "w", encoding="utf-8") as file:
        json.dump(similar_sentence, file, ensure_ascii=False, indent=4)

    return


if __name__ == "__main__":
    main()
