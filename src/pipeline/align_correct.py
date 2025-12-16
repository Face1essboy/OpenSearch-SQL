import logging
from typing import Any, Dict, List
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pipeline.utils import node_decorator, get_last_node_result, get_device
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from pipeline.utils import make_newprompt
from llm.model import model_chose
from llm.db_conclusion import *
import json
from llm.prompts import *
from runner.check_and_correct import muti_process_sql, soft_check, sql_raw_parse

@node_decorator(check_schema_status=False)
def align_correct(task: Any, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对候选SQL进行智能比对与一致性修正。

    本节点职责：
        1. 加载和整合前置节点产生的所有关键信息（库结构、候选SQL、上下文取值、fewshot等）。
        2. 对输入的候选SQL集合进行预处理，去重和标准化。
        3. 组织各类支持性prompt及数据库描述，为LLM/soft-check提供丰富上下文。
        4. 对每条SQL调用soft_check多级自动/半自动修正机制，必要时利用LLM提示/投票修正SQL错误。
        5. 统计none_case（未通过情况），输出最终修正投票结果。

    Args:
        task (Any): 当前pipeline的任务对象，包含db_id、question、evidence等。
        execution_history (List[Dict[str, Any]]): 前面各pipeline节点的执行输出历史，用于结果复用。

    Returns:
        Dict[str, Any]: {"vote": 修正及投票后的结果, "none_case": 未通过/无效情况}
    """
    # ========== 一、配置与关键资源加载 ==========
    config, node_name = PipelineManager().get_model_para()                  # 读取pipeline及节点配置
    paths = DatabaseManager()                                               # 路径管理器
    fewshot_path = paths.db_fewshot_path                                   # few-shot样例文件路径
    correct_fewshot_json = paths.db_fewshot2_path                          # 校正用few-shot样例路径
    db_sqlite_path = paths.db_path                                         # 数据库sqlite存储路径
    prompts_template = db_check_prompts()                                  # 各类LLM提示词模板
    # 从配置读取设备，如果没有则自动选择最佳设备（CUDA > MPS > CPU）
    device = get_device(config.get("device"))
    bert_model = SentenceTransformer("BAAI/bge-m3", device=device)         # 初始化句向量模型，用于软判与对齐

    # ========== 二、加载fewshot样例与修正知识库 ==========
    with open(fewshot_path, "r") as f:
        df_fewshot = json.load(f)                                          # 主要few-shot样例，按question编号索引
    with open(correct_fewshot_json, "r") as f:
        correct_dic = json.load(f)                                         # soft-check修正few-shot集

    chat_model = model_chose(node_name, config["engine"])                  # 选择当前节点对应的LLM实现

    # ========== 三、汇总各前序节点输出的关键信息 ==========
    all_db_col = get_last_node_result(execution_history, "generate_db_schema")["db_col_dic"]         # {col: [desc, ...]}
    # 列名描述字符串，用于上下文约束/提示
    column = get_last_node_result(execution_history, "column_retrieve_and_other_info")["column"]
    # 数据库外键信息
    foreign_keys = get_last_node_result(execution_history, "column_retrieve_and_other_info")["foreign_keys"]
    # 外键列集合（用于辅助修正歧义）
    foreign_set = get_last_node_result(execution_history, "column_retrieve_and_other_info")["foreign_set"]
    # 检索出的最相关列及其对应的值 [ (col, value), ... ]
    L_values = get_last_node_result(execution_history, "column_retrieve_and_other_info")["L_values"]
    # LLM判断的SELECT顺序信息（用于对齐输出字段/用户顺序需求等）
    q_order = get_last_node_result(execution_history, "column_retrieve_and_other_info")["q_order"]
    # 当前问题（如有重写则用LLM重写后的问题）
    question = get_last_node_result(execution_history, "candidate_generate")["rewrite_question"]

    # 候选SQL集合（列表形式）
    SQLs = get_last_node_result(execution_history, "candidate_generate")["SQL"]

    db = task.db_id           # 当前库名/数据库id
    hint = task.evidence      # 支持性证据
    foreign_set = set(foreign_set)      # 外键集合转set便于查重
    fewshot = df_fewshot["questions"][task.question_id]['prompt']    # 当前问题对应的few-shot
    # values转换为格式化字符串，便于prompt约束
    values = [f"{x[0]}: '{x[1]}'" for x in L_values]
    key_col_des = "#Values in Database:\n" + '\n'.join(values)       # 取值样式提示文本
    # 新db描述，嵌入表信息/外键信息，丰富上下文
    new_db_info = (f"Database Management System: SQLite\n"
                   f"#Database name: {db} \n"
                   f"{column}\n\n"
                   f"#Forigen keys:\n{foreign_keys}\n")

    db_col = {x: all_db_col[x][0] for x in all_db_col}   # {col: desc...}

    # ========== 四、候选SQL标准化与统计 ==========
    # 对全部SQL列表去特殊格式、注释等，提取纯SQL文本
    SQLs = [sql_raw_parse(x, False)[0] for x in SQLs]

    # 将候选SQL组成统计字典（去重，多次生成同一SQL计数）
    SQLs_dic = {}
    for sql in SQLs:
        SQLs_dic.setdefault(sql, 0)
        SQLs_dic[sql] += 1

    # ========== 五、prompt与soft-check管道组织 ==========
    # compose临时prompt，融合fewshot、上下文、select顺序等
    tmp_prompt = make_newprompt(prompts_template.tmp_prompt, fewshot,
                                key_col_des, new_db_info, question,
                                task.evidence, q_order)
    # 初始化动态修正检查管道（内含embedding/LLM/correct/vote所有表达式）
    Dcheck = soft_check(
        bert_model,
        chat_model,
        prompts_template.soft_prompt,
        correct_dic,
        prompts_template.correct_prompt,
        prompts_template.vote_prompt
    )

    # ========== 六、实际修正与vote环节 ==========
    # muti_process_sql负责全流程SQL修正、软/硬一致性比对与投票裁决
    vote, none_case = muti_process_sql(
        Dcheck,
        SQLs_dic,
        L_values,
        values,
        question,
        new_db_info,
        hint,
        key_col_des,
        tmp_prompt,
        db_col,
        foreign_set,
        config['align_methods'],
        db_sqlite_path,
        n=config['n']
    )

    # ========== 七、汇总输出 ==========
    response = {
        "vote": vote,           # 投票/多数判定修正输出
        "none_case": none_case  # 未通过/失败情况（如都有严重问题，便于后续定位）
    }

    return response

