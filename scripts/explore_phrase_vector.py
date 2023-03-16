import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # word2vec
    with open("./data/similar_word.json", encoding='utf-8') as file:
        similar_word = json.load(file)

    word2vec_score = [score for similar_dict in similar_word.values() for score in list(similar_dict.values())[:-1]]

    df_describe = pd.DataFrame(np.array(word2vec_score))
    print("word2vec statistic", df_describe.describe())
    print("< 0.6 percentage: ", np.mean(np.array(word2vec_score) < 0.6))

    # clip
    with open("./data/similar_sentence.json", encoding='utf-8') as file:
        similar_sentence = json.load(file)

    clip_score = [score for similar_dict in similar_sentence.values() for score in list(similar_dict.values())[:-1]]
    df_describe = pd.DataFrame(np.array(clip_score))
    print("CLIP statistic", df_describe.describe())
    print("< 0.6 percentage: ", np.mean(np.array(clip_score) < 0.6))

    # gpt
    with open("./data/similar_gpt.json", encoding='utf-8') as file:
        similar_sentence = json.load(file)

    gpt_score = [score for similar_dict in similar_sentence.values() for score in list(similar_dict.values())[:-1]]
    df_describe = pd.DataFrame(np.array(gpt_score))
    print("GPT statistic", df_describe.describe())
    print("< 0.6 percentage: ", np.mean(np.array(gpt_score) < 0.6))

    # plot
    plt.style.use('ggplot')
    my_dpi = 151
    # plt.figure(1, figsize=(3840/my_dpi, 2160/my_dpi), dpi=my_dpi)
    plt.figure(1, figsize=(20, 12), dpi=my_dpi)

    plt.hist(gpt_score, bins=None, range=None, density=None, cumulative=False, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, color=None, label="gpt", stacked=False, alpha=0.5)

    plt.hist(word2vec_score, bins=None, range=None, density=None, cumulative=False, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, color=None, label="word2vec", stacked=False, alpha=0.7)

    plt.hist(clip_score, bins=None, range=None, density=None, cumulative=False, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, color=None, label="clip", stacked=False, alpha=0.7)

    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Similarity Score', fontdict={"fontsize": 20})
    plt.ylabel('Count', fontdict={"fontsize": 20})
    plt.savefig("./data/similarity_score_histogram.jpg")

    return


if __name__ == "__main__":
    main()
