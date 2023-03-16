import json
from itertools import product

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, OpenAIGPTModel

from datasets.hico_text_label import hico_text_label


def main():
    # config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = "openai-gpt"
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
        # TODO: remove self
        if len(verb_choice) > 0 and len(obj_choice) > 0:
            synth_pair_pool.extend(synth)
        real2synth[(verb_id, obj_id)] = [(verb2id[v], obj2id[o]) for v, o in synth]

    print(len(synth_pair_pool))

    tokenizer = AutoTokenizer.from_pretrained(clip_model)
    model_proj = OpenAIGPTModel.from_pretrained(clip_model)
    model_proj.to(device)

    text_tensor = []
    pair2tensor_id = {}
    pair2sentence = {}
    idx = 0
    for i, (verb, obj) in enumerate(synth_pair_pool):
        pair = (verb2id[verb], obj2id[obj])
        if pair in pair2tensor_id:
            continue
        pair2tensor_id[pair] = idx
        idx += 1

        text = f"person {verb.replace('no_interaction', 'and').replace('_', ' ')} {obj}"
        # print(text)
        pair2sentence[pair] = text

        # inputs = tokenizer([f"person {verb.replace('no_interaction', 'and')}  {obj}"], return_tensors="pt")
        inputs = tokenizer([text], return_tensors="pt")
        inputs.to(device)

        outputs = model_proj(**inputs)
        # text_tensor.append(outputs.text_embeds)
        last_hidden_states = outputs.last_hidden_state
        # print(last_hidden_states.size(1) > 3)
        if last_hidden_states.size(1) > 3:
            # print(">3")
            text_embeds_mid = torch.mean(last_hidden_states[:, 2:-1, :], dim=1)
            # print(text_embeds_mid.size())
            # text_embeds = last_hidden_states[:, [0, -1], :]
            text_embeds = torch.cat([last_hidden_states[:, 0, :], text_embeds_mid, last_hidden_states[:, -1, :]]).\
                view(1, -1)
            # print(text_embeds.size())
            # print(text_embeds.size())
        else:
            # text_embeds = last_hidden_states[:, :3, :].view(1, -1)
            text_embeds = last_hidden_states.view(1, -1)
        # print(last_hidden_states.size())
        # print(text_embeds.size())
        print(i)
        text_tensor.append(text_embeds)

    text_tensor = torch.cat(text_tensor)
    print(text_tensor.size())

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

    text_embedding_norm = F.normalize(text_tensor, dim=-1)
    sim_matrix = torch.mm(text_embedding_norm, text_embedding_norm.t())

    print(len(pair2tensor_id))
    similar_sentence = {
        pair2sentence[k]: {
            pair2sentence[synth]: sim_matrix[pair2tensor_id[k], pair2tensor_id[synth]].item()
            for synth in v
        }
        for k, v in real2synth.items()
    }

    with open("./data/similar_gpt.json", "w", encoding="utf-8") as file:
        json.dump(similar_sentence, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
