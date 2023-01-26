import json
import torch


def main():
    with open("./checkpoint/p_202301200154_test/results.json") as file:
        result = json.load(file)
    print(len(result['preds'][1]['predictions']))


if __name__ == '__main__':
    main()
