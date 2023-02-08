import json
from itertools import product

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

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


def main():
    # config
    clip_model = "openai/clip-vit-base-patch32"
    special_sentence = "A photo of a person."
    template = "A photo of a person verb obj."
    # Load verb choice
    with open("./data/similar_word/similar_verb_google.json", encoding='utf-8') as file:
        sim_verb = json.load(file)

    # Load obj choice
    with open("./data/similar_word/similar_obj_google.json", encoding='utf-8') as file:
        sim_obj = json.load(file)

    synth_pair = []
    for verb, alter_verb in sim_verb.items():
        alter_verb = [k for k, v in alter_verb.items() if v > 0.8]
        for obj, alter_obj in sim_obj.items():
            synth_pair.append((verb, obj))  # real
            alter_obj = [k for k, v in alter_obj.items() if v > 0.8]
            prods = list(product(alter_verb, alter_obj))
            synth_pair.extend(prods)

    print(len(synth_pair))  # 29804
    # text2tensor = {}
    # tokenizer = CLIPTokenizer.from_pretrained(clip_model)
    # model_proj = CLIPTextModelWithProjection.from_pretrained(clip_model)
    # print("hoi_clip_embedding.pth is not given, using clip model to encoding.")
    # for i, text in enumerate(text2pair.keys()):
    #     inputs = tokenizer([text], padding=True, return_tensors="pt", max_length=13)
    #     outputs = model_proj(**inputs)
    #     text2tensor[text] = outputs.text_embeds
    #     # text2tensor[text] = F.log_softmax(outputs.text_embeds, dim=1)
    # torch.save(text2tensor, "./checkpoint/hoi_clip_embedding.pth")


if __name__ == '__main__':
    main()
