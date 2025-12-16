import logging
from typing import Any, Dict, List
from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from pipeline.utils import make_newprompt
from llm.model import model_chose
from llm.db_conclusion import *
import json
from llm.prompts import *
from runner.check_and_correct import get_sql

@node_decorator(check_schema_status=False)
def candidate_generate(task: Any, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    生成SQL候选集节点：
    1. 读取few-shot样例（用于prompt构建）。
    2. 整合上游节点（column、外键、取值、select顺序）信息。
    3. 使用make_newprompt构建最终prompt。
    4. 调用LLM生成SQL候选，并可带返回重写问题。
    5. 若LLM返回失败则报错；若重写问题则替换原问题。
    6. 返回结构体{"rewrite_question":..., "SQL":...}
    """
    # 获取运行配置与当前节点名称
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()
    fewshot_path = paths.db_fewshot_path

    # ===== 1. 载入few-shot样例（用于辅助提示，提高SQL准确率） =====
    with open(fewshot_path, 'r') as f:
        df_fewshot = json.load(f)

    # ===== 2. 初始化LLM与准备上下文信息 =====
    chat_model = model_chose(node_name, config["engine"])
    # 获取前置节点输出的关键信息
    column = get_last_node_result(execution_history, "column_retrieve_and_other_info")["column"]
    foreign_keys = get_last_node_result(execution_history, "column_retrieve_and_other_info")["foreign_keys"]
    L_values = get_last_node_result(execution_history, "column_retrieve_and_other_info")["L_values"]
    q_order = get_last_node_result(execution_history, "column_retrieve_and_other_info")["q_order"]
    values = [f"{x[0]}: '{x[1]}'" for x in L_values]
    db = task.db_id

    # ===== 3. 组织关键信息为prompt片段 =====
    # 用于提示LLM已知的数据库取值内容
    key_col_des = "#Values in Database:\n" + '\n'.join(values)
    # 数据库结构和外键信息
    new_db_info = (
        f"Database Management System: SQLite\n"
        f"#Database name: {db} \n"
        f"{column}\n\n"
        f"#Forigen keys:\n{foreign_keys}\n"
    )

    # 获取当前question字符串与fewshot（风格模板）
    # 可选：对问题做重写增强，一般保留原问题
    # question = rewrite_question(task.question)  
    question = task.question

    fewshot = df_fewshot["questions"][task.question_id]['prompt']
    # fewshot部分可根据配置裁剪：如仅用前半部分、或全用

    # ========== 4. 构造prompt，确保内容充实且格式清晰 ==========
    new_prompt = make_newprompt(
        db_check_prompts().new_prompt,  # 主体prompt模板
        fewshot,                       # few-shot 内容
        key_col_des,                   # 取值描述（上下文约束）
        new_db_info,                   # 结构和关系描述
        question,                      # 目标问句
        task.evidence,                 # 支持性证据或推理链
        q_order                        # select顺序或select意向词
    )

    # ========== 5. 调用LLM生成SQL候选 ==========
    # 控制是否“一问一答”模式（single=True为单步多条/False为多步模式）
    # 从字符串配置安全转换为布尔型
    single = config['single'].lower() == 'true'
    return_question = config['return_question'] == 'true'

    # SQL生成，同时可选返回LLM重写问题
    SQL, rewrite_q = get_sql(
        chat_model,
        new_prompt,
        config['temperature'],
        return_question=return_question,
        n=config['n'],
        single=single
    )

    # ===== 6. 错误兜底与问题重写逻辑 =====
    # 若LLM未返回内容，抛异常提示外部排查
    if SQL is None:
        raise ValueError("LLM API call failed in candidate_generate. API returned None.")

    # 如果重写问题不为空，则替换，否则用原问
    if rewrite_q:
        question = rewrite_q

    # ===== 7. 输出最终结构 =====
    response = {
        "rewrite_question": question,
        "SQL": SQL
        # 可选开放new_prompt便于调试
        # "new_prompt": new_prompt
    }

    return response

def rewrite_question(question):
    """
    对于涉及除法/浮点显示问题等的原问题，在尾部补充精度提示，避免SQL丢失CAST。
    """
    if question.find(" / ") != -1:
        question += ". For division operations, use CAST xxx AS REAL to ensure precise decimal results"
    return question
