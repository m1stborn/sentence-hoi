import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    plt.style.use('ggplot')
    with open("../data/similar_word.json", encoding='utf-8') as file:
        similar_word = json.load(file)

    # similarity_score_mean = [np.mean([score for score in list(similar_dict.values())])
    #                          for similar_dict in similar_word.values()]

    all_score = [score for similar_dict in similar_word.values() for score in list(similar_dict.values())[:-1]]

    df_describe = pd.DataFrame(np.array(all_score))
    print("word2vec statistic", df_describe.describe())
    print("< 0.7 percentage: ", np.mean(np.array(all_score) < 0.6))

    # plt.figure()
    plt.hist(all_score, bins=None, range=None, density=None, cumulative=False, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, color=None, label="word2vec", stacked=False, alpha=0.5)
    # plt.savefig("./data/word2vec.jpg")

    with open("../data/similar_sentence.json", encoding='utf-8') as file:
        similar_sentence = json.load(file)

    # similarity_score_mean = [np.mean([score for score in list(similar_dict.values())])
    #                          for similar_dict in similar_sentence.values()]

    sentence_score = [score for similar_dict in similar_sentence.values() for score in list(similar_dict.values())[:-1]]
    df_describe = pd.DataFrame(np.array(sentence_score))
    print("CLIP statistic", df_describe.describe())
    print("< 0.7 percentage: ", np.mean(np.array(sentence_score) < 0.6))
    # plt.figure()
    plt.hist(sentence_score, bins=None, range=None, density=None, cumulative=False, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, color=None, label="clip", stacked=False, alpha=0.5)

    plt.legend(fontsize=12)
    # plt.title("")
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.savefig("./data/similarity_score_histogram.jpg")

    return


if __name__ == "__main__":
    main()
