import json
import os

import pandas as pd


def main():
    ckpt_dir = "./checkpoint"
    for file_dir in os.listdir(ckpt_dir):
        if os.path.isdir(os.path.join(ckpt_dir, file_dir)) and file_dir != "best":
            data = []
            with open(os.path.join(ckpt_dir, file_dir, "log.txt"), encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))

            csv_filename = os.path.join(ckpt_dir, file_dir, "readable_log.csv")
            if os.path.exists(csv_filename):
                df = pd.read_csv(csv_filename)
                if len(data) <= len(df):
                    continue
            print(f"Update {csv_filename}.")
            df = pd.DataFrame(data)
            df.to_csv(csv_filename, index=False)

            # break
    # print(data)


if __name__ == '__main__':
    main()
