import logging,re
from typing import Any, Dict
from pathlib import Path
from pipeline.utils import node_decorator,get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from llm.model import model_chose
from llm.prompts import *

@node_decorator(check_schema_status=False)
def extract_query_noun(task: Any,execution_history: Dict[str, Any]) -> Dict[str, Any]:
    config,node_name=PipelineManager().get_model_para()

    chat_model = model_chose(node_name,config["engine"])
    key_col_des_raw = get_last_node_result(execution_history, "extract_col_value")["key_col_des_raw"]
    noun_ext = chat_model.get_ans(db_check_prompts().noun_prompt.format(raw_question=task.question),temperature=config["temperature"])
    values, col = parse_des(key_col_des_raw, noun_ext, debug=False)
    
    response = {
        "values":values,
        "col":col
    }
    return response

# [关键注释] — 解析从 LLM 返回的信息中提取出的 columns 和 values，并结合名词短语进行增强
def parse_des(pre_col_values, nouns, debug):
    # 去除多余注释和前缀，仅保留关键信息
    pre_col_values = pre_col_values.split("/*")[0].strip()
    if debug:
        print(pre_col_values)
    # 提取 "columns" 和 "values" 两个关键片段
    col, values = pre_col_values.split('#values:')
    _, col = col.split("#columns:")
    col = strip_char(col)
    values = strip_char(values)

    # [关键注释] — values 若为空，初始化为空列表，否则用正则提取全部引号包裹内容
    if values == '':
        values = []
    else:
        values = re.findall(r"([\"'])(.*?)\1", values)

    # [关键注释] — 提取所有名词及短语，并与原有 values 合并去重
    nouns_all = re.findall(r"([\"'])(.*?)\1", nouns)
    values_noun = set(values).union(set(nouns_all))
    values_noun = [x[1] for x in values_noun]

    # [关键注释] — 返回扩展后的 values_noun 与抽取出的 column 字段
    return values_noun, col

# [关键注释] — 简单字符串清理，去除换行和常见无关字符
def strip_char(s):
    return s.strip('\n {}[]')