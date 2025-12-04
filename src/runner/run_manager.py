import os
import json
from pathlib import Path
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple

from runner.logger import Logger
from runner.task import Task
from runner.database_manager import DatabaseManager
from runner.statistics_manager import StatisticsManager
from pipeline.workflow_builder import build_pipeline
from pipeline.pipeline_manager import PipelineManager

# 初始化 RunManager
#     ↓
# initialize_tasks() → 加载数据集，创建 Task 列表
#     ↓
# run_tasks() → 遍历所有任务
#     ↓
# worker() → 对每个任务：
#     ├─ 初始化数据库管理器
#     ├─ 创建日志记录器
#     ├─ load_checkpoint() → 加载断点（如启用）
#     ├─ build_pipeline() → 构建工作流
#     ├─ app.stream() → 执行 pipeline
#     └─ 返回最终状态
#     ↓
# task_done() → 处理结果
#     ├─ 提取评估结果
#     ├─ 更新统计信息
#     └─ 打印进度条
#     ↓
# generate_sql_files() → （可选）批量提取 SQL 文件

NUM_WORKERS = 3   # 进程池的并发进程数

# 管理任务执行流程
# 组织结果输出
# 统计和进度跟踪
# 支持断点续跑
class RunManager:
    RESULT_ROOT_PATH = "results"  # 结果输出的根目录

    def __init__(self, args: Any):
        # 初始化RunManager对象，存储传入参数和各模块的实例
        self.args = args
        self.result_directory = self.get_result_directory()  # 结果输出目录
        self.statistics_manager = StatisticsManager(self.result_directory)  # 统计结果管理器
        self.tasks: List[Task] = []  # 任务列表
        self.total_number_of_tasks = 0  # 总任务数
        self.processed_tasks = 0  # 已处理任务计数

    def get_result_directory(self) -> str:
        """
        创建并返回结果目录路径，根据输入参数自动命名，本次运行的所有输出都保存在该路径下。
        
        Returns:
            str: 结果目录的绝对路径
        """
        data_mode = self.args.data_mode  # dev/train等模式
        pipeline_nodes = self.args.pipeline_nodes  # 工作流节点组合字符串
        dataset_name = Path(self.args.db_root_path).stem  # 数据集名（目录名）
        run_folder_name = str(self.args.run_start_time)  # 当前运行开始时间
        run_folder_path = Path(self.RESULT_ROOT_PATH) / data_mode / pipeline_nodes / dataset_name / run_folder_name
        
        run_folder_path.mkdir(parents=True, exist_ok=True)  # 递归创建路径
        
        # 保存本次运行参数到 -args.json
        arg_file_path = run_folder_path / "-args.json"
        with arg_file_path.open('w') as file:
            json.dump(vars(self.args), file, indent=4)
        
        # 日志文件夹
        log_folder_path = run_folder_path / "logs"
        log_folder_path.mkdir(exist_ok=True)
        
        return str(run_folder_path)

    def initialize_tasks(self, start, end, dataset: List[Dict[str, Any]]):
        """
        从数据集初始化任务，将[start, end)区间内每条数据封装为Task。
        
        Args:
            dataset (List[Dict[str, Any]]): 数据集内容（每行是一个问题/任务）
        """
        for i, data in enumerate(dataset):
            if i < start:  # 跳过起始位置前的数据
                continue
            if i >= end:  # 超过结束位置则停止
                break
            if "question_id" not in data:
                # 若无question_id字段，则用数据下标作为id
                data = {"question_id": i, **data}
            task = Task(data)
            self.tasks.append(task)
        self.total_number_of_tasks = len(self.tasks)  # 实际任务数
        print(f"Total number of tasks: {self.total_number_of_tasks}")

    def run_tasks(self):
        """
        顺序运行所有初始化的任务。
        可以选择并行实现（已注释掉）。
        """
        # 并行实现：使用多进程池，可提升效率（适用于无全局状态干扰时）
        # with Pool(NUM_WORKERS) as pool:
        #     for task in self.tasks:
        #         pool.apply_async(self.worker, args=(task,), callback=self.task_done)
        #     pool.close()
        #     pool.join()
        for task in self.tasks:
            ans = self.worker(task)
            self.task_done(ans)

# 每个任务的工作流程：
# 初始化数据库管理器（数据库连接和缓存）
# 创建独立日志记录器
# 加载断点（如果启用）
# 构建 pipeline（根据 pipeline_nodes 配置）
# 执行 pipeline 流式处理
# 返回最终状态
    def worker(self, task: Task) -> Tuple[Any, str, int]:
        """
        单个任务的执行流程（一个task）。
        
        Args:
            task (Task): 待处理任务
        
        Returns:
            tuple: 任务最终状态对象、数据库ID、问题ID
        """
        # 注意：本函数用于进程池或顺序遍历均可
        # 数据库管理器（单例模式，负责数据库缓存等）
        database_manager = DatabaseManager(db_mode=self.args.data_mode, db_root_path=self.args.db_root_path, db_id=task.db_id)
        # 日志对象，每题有独立日志
        logger = Logger(db_id=task.db_id, question_id=task.question_id, result_directory=self.result_directory)
        logger._set_log_level(self.args.log_level)
        logger.log(f"Processing task: {task.db_id} {task.question_id}", "info")
        # PipelineManager负责封装pipeline各环节的超参数配置
        pipeline_manager = PipelineManager(json.loads(self.args.pipeline_setup))
        # 加载断点/历史执行轨迹用于断点续跑（如有）
        execution_history = self.load_checkpoint(task.db_id, task.question_id)

        # pipeline初始输入状态（内部键值约定）
        initial_state = {"keys": {"task": task, "execution_history": execution_history}}
        print("Building pipeline...")
        self.app = build_pipeline(self.args.pipeline_nodes)  # 构建流程活性图
        print("Pipeline built successfully.")

        # 取出pipeline最后一个结点名，用于结果定位
        if hasattr(self.app, 'nodes'):
            # 获取最后一个节点的键
            if self.app.nodes:  # 确保节点字典非空
                last_node_key = list(self.app.nodes.keys())[-1]  # 最后一个节点类型名
                print('checkpoint final: ', last_node_key)
            else:
                last_node_key = None  # 没有节点，置空
        else:
            last_node_key = None  # 没有nodes属性

        # 按流水线流程顺序迭代（实际流式、可能带yield/step出错处理）
        for state in self.app.stream(initial_state):
            continue  # 通常仅执行循环，实际结果保存在state

        # 返回最终状态（最后一个节点）、任务标号
        return state[last_node_key], task.db_id, task.question_id
        # 异常处理代码块可打开，便于debug
        # except Exception as e:
        #     logger.log(f"Error processing task: {task.db_id} {task.question_id}\n{e}", "error")
        #     return None, task.db_id, task.question_id

# 从最终状态提取评估结果
# 更新统计信息并保存
# 更新进度条
    def task_done(self, log: Tuple[Any, str, int]):
        """
        每个任务完成后的回调函数，负责统计与进度条打印。
        
        Args:
            log (tuple): 任务执行结果组成的三元组
        """
        state, db_id, question_id = log
        # print('-'*20)
        # print(state)
        if state is None:
            return
        # 取执行历史中的最后一步（通常是evaluation节点）
        evaluation_result = state["keys"]['execution_history'][-1]
        if evaluation_result.get("node_type") == "evaluation":
            # 遍历评价项进行累计统计
            for evaluation_for, result in evaluation_result.items():
                if evaluation_for in ['node_type', 'status']:
                    continue  # 跳过类型和状态字段
                self.statistics_manager.update_stats(db_id, question_id, evaluation_for, result)
            self.statistics_manager.dump_statistics_to_file()  # 保存统计文件
        self.processed_tasks += 1
        self.plot_progress()  # 打印进度条

    def plot_progress(self, bar_length: int = 100):
        """
        进度条打印。每处理一题更新一次。
        
        Args:
            bar_length (int, optional): 进度条总长度，默认100
        """
        processed_ratio = self.processed_tasks / self.total_number_of_tasks
        progress_length = int(processed_ratio * bar_length)
        # 逐行刷新（覆盖前一进度条行，适用于大部分终端）
        print('\x1b[1A' + '\x1b[2K' + '\x1b[1A')
        print(f"[{'=' * progress_length}>{' ' * (bar_length - progress_length)}] {self.processed_tasks}/{self.total_number_of_tasks}")

    def load_checkpoint(self, db_id, question_id) -> Dict[str, List[str]]:
        """
        尝试加载断点续跑文件，将历史执行记录恢复到内存。
        仅当self.args.use_checkpoint为True时有效。
        
        Args:
            db_id: 数据库ID
            question_id: 问题ID
        Returns:
            执行步骤列表（每步为一字典）
        """
        # tentative_schema = DatabaseManager().get_db_schema()
        execution_history = []
        if self.args.use_checkpoint:
            checkpoint_file = Path(self.args.checkpoint_dir) / f"{question_id}_{db_id}.json"
            print(checkpoint_file)
            if checkpoint_file.exists():
                with checkpoint_file.open('r') as file:
                    checkpoint = json.load(file)
                    for step in checkpoint:
                        node_type = step["node_type"]
                        if node_type in self.args.checkpoint_nodes:
                            execution_history.append(step)
                            # print(execution_history)
                        # if "tentative_schema" in step:
                            # tentative_schema = step["tentative_schema"]
            else:
                Logger().log(f"Checkpoint file not found: {checkpoint_file}", "warning")
            # 打印断点恢复到的最后步骤
            print("checkpoint end: ", execution_history[-1]["node_type"])
        return execution_history

# 后处理：从执行历史 JSON 中提取各节点类型的 SQL
# 按节点类型组织并保存为独立 JSON 文件
    def generate_sql_files(self):
        """
        批量生成所有节点类型的SQL保存文件。将每个节点类型对应的SQL按question_id归档到json。
        """
        sqls = {}  # {node_type:{qid:sql,...}, ...}
        
        # 遍历result目录下所有json文件，将SQL抽取出来封装到dict
        for file in os.listdir(self.result_directory):
            if file.endswith(".json") and "_" in file:
                _index = file.find("_")
                question_id = int(file[:_index])
                db_id = file[_index + 1:-5]
                with open(os.path.join(self.result_directory, file), 'r') as f:
                    exec_history = json.load(f)
                    for step in exec_history:
                        if "SQL" in step:
                            node_type = step["node_type"]
                            if node_type not in sqls:
                                sqls[node_type] = {}
                            sqls[node_type][question_id] = step["SQL"]
        # Save to json by node_type
        for key, value in sqls.items():
            with open(os.path.join(self.result_directory, f"-{key}.json"), 'w') as f:
                json.dump(value, f, indent=4, ensure_ascii=False)

