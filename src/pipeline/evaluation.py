import logging
from typing import Dict, Any

from runner.logger import Logger
from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result
from runner.check_and_correct import sql_raw_parse

@node_decorator(check_schema_status=False)
def evaluation(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    评估节点：对比不同阶段生成的SQL与标准答案（ground truth SQL），并执行评测。
    本函数针对如下阶段的输出评估其SQL与标准SQL的执行情况：
        - candidate_generate: 原始候选SQL
        - align_correct: 对齐纠错后的SQL
        - vote: 投票融合后的SQL

    参数:
        task (Any): 当前任务对象，至少含标准SQL（SQL）、原始问题(raw_question)、证据(evidence)等信息。
        execution_history (Dict[str, Any]): 该任务前序节点（如candidate_generate/align_correct/vote）的所有执行产物。

    返回:
        Dict[str, Any]: 每种生成/修正方式下的SQL与执行结果信息字典。
    """
    # 获取标准SQL，作为基准（GOLD SQL）
    ground_truth_sql = task.SQL

    # 定义要评估的各阶段节点及其输出：
    # - candidate_generate: 原始生成SQL列表
    # - align_correct: 纠错/对齐后的SQL投票列表
    # - vote: 最终投票/融合的SQL
    to_evaluate = {
        "candidate_generate": get_last_node_result(execution_history, "candidate_generate"), 
        "align_correct": get_last_node_result(execution_history, "align_correct"), # 对齐+纠错后输出
        # 下面两行为其它可能阶段，暂未启用:
        # "align": get_last_node_result(execution_history, "vote"), # 未纠错时（如需可补充）
        # "correct": get_last_node_result(execution_history, "vote"),
        "vote": get_last_node_result(execution_history, "vote") # 投票最终输出
    }

    result = {}  # 汇总最终评测结果
    for evaluation_for, node_result in to_evaluate.items():
        predicted_sql = "--"  # 初始化预测SQL，兜底填充为注释
        evaluation_result = {}  # 记录执行/评测结果

        try:
            # 仅节点成功生成时（如status==success）才进行比对或执行
            if node_result.get("status", "") == "success":
                # 按阶段/节点类型分别提取其预测SQL，保证来源清晰
                if evaluation_for == "align":
                    # 若启用align阶段，采用其 SQL_align_vote 字段
                    predicted_sql = node_result.get('SQL_align_vote', "--")
                elif evaluation_for == "correct":
                    # 若启用correct阶段，采用其 SQL_correct_vote 字段
                    predicted_sql = node_result.get("SQL_correct_vote", "--")
                elif evaluation_for == "align_correct":
                    # align_correct一般为纠错投票后的SQL集合，取首个候选
                    vote_all = node_result['vote']
                    predicted_sql = vote_all[0]['sql'] if vote_all else "--"
                elif evaluation_for == "candidate_generate":
                    # 原始候选SQL列表，平滑化并取第一个
                    candidate_all = node_result['SQL']
                    predicted_sql = sql_raw_parse(candidate_all[0], False)[0] if candidate_all else "--"
                elif evaluation_for == "vote":
                    # 投票/融合阶段直接取其最终SQL
                    predicted_sql = node_result.get("SQL", "--")

                # 使用DatabaseManager对预测SQL与标准SQL进行执行/结果对比（包含容错与超时机制）
                response = DatabaseManager().compare_sqls(
                    predicted_sql=predicted_sql,
                    ground_truth_sql=ground_truth_sql,
                    meta_time_out=180
                )
                # 记录执行返回信息（结果/错误详情等）
                evaluation_result.update({
                    "exec_res": response.get("exec_res", ""),
                    "exec_err": response.get("exec_err", "")
                })
            else:
                # 节点失败时，直接记录错误信息
                evaluation_result.update({
                    "exec_res": "generation error",
                    "exec_err": node_result.get("error", "unknown error"),
                })
        except Exception as e:
            # 捕获异常，日志并记录error
            Logger().log(
                f"Node 'evaluate_sql': {task.db_id}_{task.question_id}\n{type(e)}: {e}\n",
                "error",
            )
            evaluation_result.update({
                "exec_res": "error",
                "exec_err": str(e),
            })

        # 为便于分析，补充记录题目、证据、标准与预测SQL内容
        evaluation_result.update({
            "Question": task.raw_question,
            "Evidence": task.evidence,
            "GOLD_SQL": ground_truth_sql,
            "PREDICTED_SQL": predicted_sql
        })
        # 综合结果加入每阶段评测输出
        result[evaluation_for] = evaluation_result

    # 记录评估成功信息（实际可选）
    logging.info("Evaluation completed successfully")
    return result
