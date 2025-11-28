import argparse
import json
import os
import pickle
from pathlib import Path
import sqlite3
from tqdm import tqdm
import logging
# 设置基本配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
## 为dev和train生成合适的文件，并保存raw_question字段

def bird_pre_process(
    bird_dir,
    with_evidence=False,
    dev_json="dev/dev.json",
    train_json="train/train_json",
    dev_table="dev/dev_tables.json",
    train_table="train/train_tables.json"
):
    """
    预处理bird相关数据，通过加载JSON文件，按需将evidence追加到question后面，并将处理后数据保存到指定目录。

    Args:
        bird_dir (str): 数据文件根目录
        with_evidence (bool, optional): 若为True，则把evidence添加到question后面。默认False。
        dev_json (str, optional): dev数据集JSON路径。默认"dev/dev.json"。
        train_json (str, optional): train数据集JSON路径。默认"train_json"。
        dev_table (str, optional): dev表结构JSON路径。默认"dev/dev_tables.json"。
        train_table (str, optional): train表结构JSON路径。默认"train/train_tables.json"。

    Returns:
        None: 该函数处理数据并保存到磁盘
    """
    # 定义子函数，对原始json进行预处理
    def json_preprocess(data_jsons):
        new_datas = []
        for data_json in data_jsons:
            # 保存原始问题内容
            data_json["raw_question"] = data_json['question']
            # 如需加证据，且evidence非空，则拼接evidence到question
            if with_evidence and len(data_json.get("evidence", "")) > 0:
                data_json['question'] = (data_json['question'] + " " + data_json["evidence"]).strip()
            new_datas.append(data_json)
        return new_datas

    # 预处理输出目录为根目录下的data_preprocess子目录
    preprocessed_path = Path(bird_dir) / "data_preprocess"
    preprocessed_path.mkdir(exist_ok=True)

    # 处理dev数据集，读取、预处理后保存
    with open(os.path.join(bird_dir, dev_json)) as f:
        data_jsons = json.load(f)
        wf = open(os.path.join(preprocessed_path, 'dev.json'), 'w')
        json.dump(json_preprocess(data_jsons), wf, indent=4)

    # 处理train数据集，读取、预处理后保存
    with open(os.path.join(bird_dir, train_json)) as f:
        data_jsons = json.load(f)
        wf = open(os.path.join(preprocessed_path, 'train.json'), 'w')
        json.dump(json_preprocess(data_jsons), wf, indent=4)

    # 合并dev和train的表结构信息
    tables = []
    with open(os.path.join(bird_dir, dev_table)) as f:
        tables.extend(json.load(f))
    with open(os.path.join(bird_dir, train_table)) as f:
        tables.extend(json.load(f))
    # 存表结构到统一的tables.json
    with open(os.path.join(preprocessed_path, 'tables.json'), 'w') as f:   # 共80个数据
        json.dump(tables, f, indent=4)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Process the Bird dataset.")
    parser.add_argument('--db_root_directory', type=str, help='Root directory for the database.')
    parser.add_argument('--dev_json', type=str, help='Path to the Dev JSON file.')
    parser.add_argument('--train_json', type=str, help='Path to the training JSON file.')
    parser.add_argument('--dev_table', type=str, help='Dev table name.')
    parser.add_argument('--train_table', type=str, help='Training table name.')

    args = parser.parse_args()
    logging.info(f"Start data_preprocess,the output_file is {args.db_root_directory}/data_preprocess")
    # 调用bird_pre_process进行实际的数据处理
    bird_pre_process(
        bird_dir=args.db_root_directory, 
        with_evidence=True, 
        dev_json=args.dev_json, 
        train_json=args.train_json, 
        dev_table=args.dev_table, 
        train_table=args.train_table
    )
