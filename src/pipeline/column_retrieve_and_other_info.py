import logging
import re
import json
from typing import Any, Dict
from pathlib import Path

from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from sentence_transformers import SentenceTransformer
from llm.model import model_chose
from llm.db_conclusion import find_foreign_keys_MYSQL_like
from llm.prompts import *
from runner.extract import DES_new
from database_process.make_emb import load_emb
from runner.column_retrieve import ColumnRetriever
from runner.column_update import ColumnUpdater

@node_decorator(check_schema_status=False)
def column_retrieve_and_other_info(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    该节点负责基于查询问题(task)、数据库schema和embedding向量等，检索最相关的列以及相关信息，并返回
    详细记录了各步骤的处理流程。核心逻辑如下：
    1. 初始化模型、路径、参数，加载embedding。
    2. 结合生成的schema列和名词抽取结果选出候选columns。
    3. 利用检索器与列相关性计算、多轮更新得到最终的column集合。
    4. 使用DES模型对输入值(values)与db特征进行semi-match，得到最匹配的值和值对应的columns。
    5. 查找外键，并辅助columns推理修正。
    6. 向LLM询问select最终输出格式（q_order）。
    7. 组装返回结构。
    """
    # ========================== 初始化 & 参数 ==============================
    config,node_name = PipelineManager().get_model_para()  # 获取配置以及当前节点名
    paths = DatabaseManager()                              # 路径管理器，获取 embedding 路径、schema 路径等
    emb_dir = paths.emb_dir                                # embedding 文件夹路径
    tables_info_dir = paths.db_tables                      # 表结构信息路径
    chat_model = model_chose(node_name, config["engine"])  # 选择LLM模型
    bert_model = SentenceTransformer("BAAI/bge-m3")        # 句向量模型用于列/值语义检索

    # ========================== 数据准备 ==============================
    all_db_col = get_last_node_result(execution_history, "generate_db_schema")["db_col_dic"] # 数据库所有列映射 {col: [desc, ...], ...}
    origin_col = get_last_node_result(execution_history, "extract_query_noun")["col"]        # 上一节点抽取出的column
    values = get_last_node_result(execution_history, "extract_query_noun")["values"]         # 上一节点抽取出的values

    db = task.db_id

    # ========================== 加载embedding数据 ==============================
    emb_values_dic = {}           # 缓存每个db的embedding和原始值
    if emb_values_dic.get(db):
        DB_emb, col_values = emb_values_dic[db]           # 已加载，直接获取
    else:
        DB_emb, col_values = load_emb(db, emb_dir)        # 从磁盘加载embedding向量和col原始值
        emb_values_dic[db] = [DB_emb, col_values]

    # ========================== 列相关性检索初步候选 ==============================
    # db_col: {col名: 表达式字符串（首个desc）}
    db_col = {x: all_db_col[x][0] for x in all_db_col}
    db_keys_col = all_db_col.keys()   # 所有列名集合

    # 使用ColumnRetriever模块语义检索，生成初候选column集合（考虑bert相关性）
    col_retrieve = ColumnRetriever(bert_model, tables_info_dir).get_col_retrieve(
        task.question, db, db_keys_col
    )

    # ========================== 外键信息辅助推理，更新column候选 ==============================
    # 查找所有外键及其集合，辅助主外键推理
    foreign_keys, foreign_set = find_foreign_keys_MYSQL_like(tables_info_dir, db)
    # pre_update: 利用外键信息、候选columns、上下文列(origin_col)等，对column集合进一步更新、补充
    cols = ColumnUpdater(db_col).col_pre_update(origin_col, col_retrieve, foreign_set)

    # ========================== 使用特征embedding模型, 检索最匹配列和取值 ==============================
    des = DES_new(bert_model, DB_emb, col_values)
    cols_select, L_values = des.get_key_col_des(
        cols,
        values,
        debug=False,
        topk=config['top_k'],
        shold=0.65
    )   # 根据输入values、候选列，获得最相关的列（cols_select）和匹配的值集合（L_values）

    # ========================== 列suf处理 & LLM输出列顺序 ==============================
    column = ColumnUpdater(db_col).col_suffix(cols_select)  # 对选中列再做后处理，格式化输出
    # values = [f"{x[0]}: '{x[1]}'" for x in L_values]     # 形式上的处理（可注释）

    # 查询order指令的多次重试机制
    count = 0
    while count < 3:
        try:
            # 通过select_prompt构造问题提示，LLM返回用户原问对应SQL SELECT字段的期望顺序等
            q_order = query_order(
                task.raw_question,
                chat_model,
                db_check_prompts().select_prompt,
                temperature=config['temperature']
            )
            break
        except Exception as e:
            count += 1
            if count == 3:
                logging.warning(f"query_order重试三次失败，异常为: {e}")
                q_order = []

    # ========================== 汇总返回结果 ==============================
    response = {
        # "col_retrieve": list(col_retrieve),      # 过程调试可用，正式输出可注释
        # "col_select": list(cols_select),         # 同上，仅供调试
        "L_values": L_values,                      # 检索出的最相关特征值 [(col, value), ...]
        "column": column,                          # 最终确定的columns，按select顺序
        "foreign_keys": foreign_keys,              # 外键信息（关系，约束等）
        "foreign_set": foreign_set,                # 外键集合，辅助进一步推理
        "q_order": q_order                         # LLM认为select语句应输出内容（顺序/内容等）
    }

    return response

def query_order(question, chat_model, select_prompt, temperature):
    """
    使用LLM获得问题question所需select语句中字段输出的顺序和内容，便于对齐select和用户意图。
    对返回格式做json解析。
    """
    # 用select_prompt构造提示词
    select_prompt = select_prompt.format(question=question)
    ans = chat_model.get_ans(select_prompt, temperature=temperature)
    # 去除格式化残留的markdown代码块
    ans = re.sub("```json|```", "", ans)
    # 转为json结构
    select_json = json.loads(ans)
    # 进一步抽取select内容及类型判断
    res, judge = json_ext(select_json)
    return res

def json_ext(jsonf):
    """
    解析LLM生成的SELECT内容json结构，抽取出所需字段顺序、类型等准确信息。
    支持多种问题结构（QIC, JC...)。
    """
    ans = []
    judge = False
    for x in jsonf:
        if x["Type"] == "QIC":
            Q = x["Extract"]["Q"].lower()
            # 针对不同疑问词返回内容排序
            if Q in ["how many", "how much", "which", "how often"]:
                for item in x["Extract"]["I"]:
                    ans.append(x["Extract"]["Q"] + " " + item)
            elif Q in ["when", "who", "where"]:
                ans.append(x["Extract"]["Q"])
            else:
                ans.extend(x["Extract"]["I"]) # 其他疑问词直接拼接信息项
        elif x["Type"] == "JC":
            ans.append(x["Extract"]["J"]) # 判断型（如true/false/yes/no）
            judge = True
    return ans, judge
