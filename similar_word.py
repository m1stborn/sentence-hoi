import json

import torch
from typing import Dict, List

import gensim.downloader
import numpy as np
import torch.nn.functional as F
from datasets.hico_text_label import hico_obj_text_label


def main():
    # Covert label (the annotation have different object id)
    # obj2id = {object_text.replace("a photo of an ", "").replace("a photo of a ", "").replace(" ", "_"): idx
    #           for idx, object_text in hico_obj_text_label}
    # with open('data/similar_word/obj2id.json', 'w', encoding='utf-8') as f:
    #     json.dump(obj2id, f, ensure_ascii=False, indent=4)

    # Load verb
    verb2idx = {}
    with open('data/hico_list_vb.txt', encoding='utf-8') as f:
        f.readline()
        f.readline()  # skip first 2 line
        for i, line in enumerate(f):
            idx, verb = line.split()
            verb2idx[verb] = idx

    # Load object
    obj2idx = {}
    with open('data/hico_list_obj.txt', encoding='utf-8') as f:
        f.readline()
        f.readline()  # skip 2 line
        for i, line in enumerate(f):
            idx, obj = line.split()
            obj2idx[obj] = idx

    # print(verb2idx.keys(), obj2idx.keys())

    # Using word2vec-google-news-300d pretrain
    # Load glove pretrain weight
    glove_vectors = gensim.downloader.load('word2vec-google-news-300')

    verb2sim_word: Dict[str, Dict[str, float]] = {}
    for verb in verb2idx.keys():
        # verb = verb.split("_")[0]
        original_verb = verb
        # vector = glove_vectors[verb]
        try:
            sims = glove_vectors.most_similar(verb.replace("_", ""), topn=10)  # get other similar word
        except KeyError:
            sims = glove_vectors.most_similar(verb.split("_")[0], topn=10)  # get other similar word

        verb2sim_word[original_verb] = {sim: score for sim, score in sims}

    with open("data/similar_word/similar_verb_google.json", "w", encoding="utf-8") as file:
        json.dump(verb2sim_word, file, ensure_ascii=False, indent=4)

    obj2sim_word: Dict[str, Dict[str, float]] = {}
    for obj in obj2idx.keys():
        # obj = obj.split("_")[0]
        original_obj = obj
        try:
            sims = glove_vectors.most_similar(obj.replace("_", ""), topn=10)  # get other similar word
        except KeyError:
            sims = glove_vectors.most_similar(obj.split("_")[0], topn=10)  # get other similar word

        obj2sim_word[original_obj] = {sim: score for sim, score in sims}

    with open("data/similar_word/similar_obj_google.json", "w", encoding="utf-8") as file:
        json.dump(obj2sim_word, file, ensure_ascii=False, indent=4)

    return

    # --------------------------------------------
    # Using glove6B.300d.txt
    # word2vec: Dict[str, np.ndarray] = {}
    # with open('./checkpoint/glove.6B.300d.txt', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         # if i == 1000:
    #         #     break
    #         values = line.split()
    #         word = values[0]
    #         vectors = np.asarray(values[1:], dtype='float32')
    #         word2vec[word] = vectors
    #
    # word2tensor: Dict[str, torch.Tensor] = {k: torch.from_numpy(v) for k, v in word2vec.items()}
    # print(f'Found {len(word2vec)} word vectors.')
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # text_embeddings = torch.stack(list(word2tensor.values()))  # [vocab_size, 300d]
    # text_embeddings_norm = F.normalize(text_embeddings, dim=-1)
    #
    # index2word = np.asarray([k for k in word2vec.keys()])
    # # Word similarity matrix
    # # Solution 1: out of memory(Fail)
    # # sim_matrix = torch.mm(text_embeddings_norm, text_embeddings_norm.t())  # [vocab_siz, vocab_size]
    # # top_k_index = torch.topk(sim_matrix, k=10, dim=-1).indices.cpu()  # [vocab_size, top_k]
    # #
    # # # Find top k similar word
    # # verb2sim_word: Dict[str, np.ndarray] = {k: index2word[top_k_index[i].tolist()]
    # #                                         for i, k in enumerate(verb2idx.keys())}
    # #
    # # obj2sim_word: Dict[str, np.ndarray] = {k: index2word[top_k_index[i].tolist()]
    # #                                        for i, k in enumerate(obj2idx.keys())}
    #
    # # Solution 2: calculate a word at a time
    # verb2sim_word: Dict[str, np.ndarray] = {}
    # obj2sim_word: Dict[str, np.ndarray] = {}
    #
    # # TODO: filter by score threshold
    # # TODO: handle split word
    # for verb in verb2idx.keys():
    #     verb = verb.split("_")[0]
    #     verb_tensor = word2tensor[verb].unsqueeze(0)  # torch.Size([1, 300])
    #     sim_score = torch.mm(verb_tensor, text_embeddings_norm.t())  # torch.Size([1, 400000])
    #     top_k_index = torch.topk(sim_score, k=10, dim=-1).indices.squeeze(0)  # torch.Size([10]) = [vocab_size]
    #     verb2sim_word[verb] = index2word[top_k_index.tolist()].tolist()
    #
    # with open("./data/similar_verb.json", "w", encoding="utf-8") as file:
    #     json.dump(verb2sim_word, file, ensure_ascii=False, indent=4)
    #
    # for obj in obj2idx.keys():
    #     obj = obj.split("_")[0]
    #     obj_tensor = word2tensor[obj].unsqueeze(0)  # torch.Size([1, 300])
    #     sim_score = torch.mm(obj_tensor, text_embeddings_norm.t())  # torch.Size([1, 400000])
    #     top_k_index = torch.topk(sim_score, k=10, dim=-1).indices.squeeze(0)  # torch.Size([10]) = [vocab_size]
    #     obj2sim_word[obj] = index2word[top_k_index.tolist()].tolist()
    #
    # with open("./data/similar_obj.json", "w", encoding="utf-8") as file:
    #     json.dump(obj2sim_word, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
