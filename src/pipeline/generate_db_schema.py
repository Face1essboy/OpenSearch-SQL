import logging
from typing import Any, Dict
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pipeline.utils import node_decorator
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from llm.model import model_chose
from llm.db_conclusion import *
import json
import os

@node_decorator(check_schema_status=False)
def generate_db_schema(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    该节点用于生成数据库的 schema 信息及列字典（db_col_dic），并缓存到 db_schema.json。
    主要功能：
      - 载入/初始化所需模型
      - 读取参数与路径信息
      - 尝试从缓存(db_schema.json)读取目标 db 的 schema 信息
      - 若无，则通过代理调用 get_allinfo，生成并写回缓存
      - 返回 {"db_list": all_info, "db_col_dic": db_col}
    """
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()
    # [关键注释] — 初始化所需的 bert_model
    bert_model = SentenceTransformer("BAAI/bge-m3")

    # [关键注释] — 路径参数
    db_json_dir = paths.db_json            # 数据库结构/表结构描述
    tables_info_dir = paths.db_tables      # 表信息描述
    sqllite_dir = paths.db_path            # sqlite db 路径
    db_dir = paths.db_directory_path       # 原始 db 目录路径
    chat_model = model_chose(node_name, config["engine"])  # 选择llm

    # [关键注释] — 对输出cache的统一管理与格式
    # todo 这可能是后续可以改变为M-Schema的
    ext_file = Path(paths.db_root_path) / "db_schema.json"

    # [关键注释] — 读取 schema 缓存，如果不存在则初始化空
    if os.path.exists(ext_file):
        with open(ext_file, 'r') as f:
            data = json.load(f)  # 格式: {db_id: [all_info, db_col]}
    else:
        data = {}

    # [关键注释] — 获取数据库信息代理/handler
    DB_info_agent = db_agent_string(chat_model)
    
    db = task.db_id
    existing_entry = data.get(db)
    # [关键注释] — 检查目标 db 是否已被缓存
    if existing_entry:
        all_info, db_col = existing_entry
    else:
        # [关键注释] — 缓存不存在，需现算并落盘
        # todo 修改为M-Schema
        all_info, db_col = DB_info_agent.get_allinfo(
            db_json_dir, db, sqllite_dir, db_dir, tables_info_dir, bert_model
        )
        data[db] = [all_info, db_col]
        with open(ext_file, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # [关键注释] — 主要结果（后续节点依赖该格式）
    response = {
        "db_list": all_info,     # 数据库结构信息
        "db_col_dic": db_col     # 列字典（便于后续节点 select）
    }
    return response
