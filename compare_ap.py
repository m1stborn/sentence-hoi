import json

import matplotlib.pyplot as plt
import pandas as pd


def main():
    with open("./checkpoint/p_202302211220/mAP_default.json", "r", encoding="utf-8") as file:
        result = json.load(file)
    with open("./data/annotations/hoi_id_to_num.json") as file:
        hoi_dict = json.load(file)
    with open("./data/annotations/hoi_list_new.json") as file:
        hoi_list = json.load(file)

    hoi_counts = [v['num'] for k, v in hoi_dict.items()]
    hoi_ap = [f"{v*100:.2f}" for k, v in result['AP'].items()]
    hoi_label = [k for k, v in result['AP'].items()]
    text = [f"{d['verb']} {d['object']}" for d in hoi_list]
    df = pd.DataFrame.from_dict({
        "labels": hoi_label,
        "ap": hoi_ap,
        "counts": hoi_counts,
        "text": text
    })
    df.to_csv("./assets/compare_ap.csv", index=False)
    # Plot Ap per  history
    # hoi_counts, hoi_ap, hoi_label = zip(*sorted(zip(hoi_counts, hoi_ap, hoi_label)))
    # idx = list(range(len(hoi_ap)))

    # plot
    # plt.style.use('ggplot')
    # my_dpi = 151
    # fig, ax1 = plt.subplots(figsize=(3840 / 20, 2160 / 40), dpi=my_dpi)
    # # ax1.bar(idx, hoi_counts, tick_label=hoi_label)
    # # ax1.set_xticks(idx, rotation=90, fontsize=10)
    # # bar = plt.bar(idx, hoi_counts, tick_label=hoi_label)
    # plt.yticks(fontsize=100)
    # plt.xticks(rotation=90, fontsize=10)
    # # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    # # ax2.plot(idx, hoi_ap, color='b' , scaley=True, label="ap")
    # plt.plot(idx, hoi_ap, color='b', scaley=True, label="ap")
    #
    # plt.ylabel("Count", fontsize=15)
    # plt.xlabel("Label", fontsize=15)
    # plt.savefig("./assets/ap.jpg")

    return


if __name__ == "__main__":
    main()
