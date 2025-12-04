import argparse
import json
from datetime import datetime
from typing import Any, Dict, List
from runner.run_manager import RunManager
import os

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    # 读取JSON文件，返回包含字典的列表
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def main(args):
    """
    Main function to run the pipeline with the specified configuration.
    """
    # 构建用于读取数据集的文件路径
    db_json=os.path.join(args.db_root_path,'data_preprocess',f'{args.data_mode}.json')
    
    # 加载数据集
    dataset = load_dataset(db_json)

    # 初始化RunManager并分配任务
    run_manager = RunManager(args)
    run_manager.initialize_tasks(args.start, args.end, dataset)
    # 执行任务
    run_manager.run_tasks()
    # 生成SQL文件，用于保存结果
    run_manager.generate_sql_files()

if __name__ == '__main__':
    # 解析命令行参数
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data_mode', type=str, required=True, help="Mode of the data to be processed.")
    args_parser.add_argument('--db_root_path', type=str, required=True, help="Path to the data file.")
    args_parser.add_argument('--pipeline_nodes', type=str, required=True, help="Pipeline nodes configuration.")
    args_parser.add_argument('--pipeline_setup', type=str, required=True, help="Pipeline setup in JSON format.")
    args_parser.add_argument('--use_checkpoint', action='store_true', help="Flag to use checkpointing.")
    args_parser.add_argument('--checkpoint_nodes', type=str, required=False, help="Checkpoint nodes configuration.")
    args_parser.add_argument('--checkpoint_dir', type=str, required=False, help="Directory for checkpoints.")
    args_parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    args_parser.add_argument('--start', type=int, default=0, help="Start point")
    args_parser.add_argument('--end', type=int, default=1, help="End point")
    # 解析参数
    args = args_parser.parse_args()
    # 记录当前运行开始时间，用于结果文件夹命名
    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # 检查是否使用断点续跑(checkpoint)功能，若是则需补充相关参数校验
    if args.use_checkpoint:
        print('Using checkpoint')
        if not args.checkpoint_nodes:
            raise ValueError('Please provide the checkpoint nodes to use checkpoint')
        if not args.checkpoint_dir:
            raise ValueError('Please provide the checkpoint path to use checkpoint')
    
    # 程序主入口，开始执行
    main(args)
