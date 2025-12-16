import logging
from typing import Any, Dict
from pathlib import Path
from pipeline.utils import node_decorator, get_last_node_result
from runner.check_and_correct import sql_raw_parse

def vote_single(vote_all, mod="answer", SQLs=[]):
    """
    进行SQL候选答案的投票选择。

    Args:
        vote_all: List[Dict]，每个元素为一条SQL候选及其执行统计（答案、计数、耗时等）。
        mod: str，指从候选结构中选取哪个字段作为比对内容（'answer'/'correct_ans'等）。
        SQLs: List[str]，原始SQL字符串列表，作为兜底输出。

    Returns:
        ans: 最终选出的SQL语句（多数派、耗时最短原则）。
        maxm: 获得最高票数（最多答案重合）的票数。
        min_t: 获胜SQL的最短耗时。
        vote_M: List[int]，每个候选获得的票数。
    """
    vote_M = [0] * len(vote_all)  # 每个候选的票数统计
    same_ans = {}                 # 记录答案重合的代表组
    for i, item in enumerate(vote_all):
        sql = item["sql"]
        ans = item[mod]              # 当前候选的答案（如执行返回值的集合）
        count = item["count"]        # 该候选原始出现次数
        time_cost = item["time_cost"]

        # 累积自身初始票数
        vote_M[i] += count

        # 初始化同组标记（并查集思想-每组代表自身下标）
        if same_ans.get(i, -1) == -1:
            same_ans[i] = i
        if not ans:
            # 如无答案，票数直接归零，不参与投票
            vote_M[i] = 0
            continue
        # 与之后所有候选进行一一比对
        for j in range(i + 1, len(vote_all)):
            other_ans = vote_all[j][mod]  # 另一个候选的答案
            # 精确比较，只有完全相等(可按需求改为集合包含等)才算同答案
            if ans == other_ans:
                same_ans[j] = same_ans[i]
                # 两者票数互相加和
                vote_M[i] += vote_all[j]["count"]
                vote_M[j] += vote_all[i]["count"]
            # 下面注释是更早的旧实现对比字段和索引方式，已不用

    maxm = max(vote_M)  # 最大的票数
    min_t = 1_000_000   # 初始化极大耗时
    # 默认设为兜底SQL（如无投票命中，回退第一个SQL串）
    sql_0 = sql_raw_parse(SQLs[0], False)[0]
    ans = sql_0

    print("_______vote same best", same_ans)
    # 选择获得最高票数且耗时最短的SQL
    for i, x in enumerate(vote_M):
        if maxm == x:
            if vote_all[i]["time_cost"] < min_t:
                ans = vote_all[i]['sql']
                min_t = vote_all[i]["time_cost"]
    return ans, maxm, min_t, vote_M
      


@node_decorator(check_schema_status=False)
def vote(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    投票融合节点，对来自前一节点的多个SQL候选结构，按照答案一致性和票数进行多数派表决。

    Args:
        task: 当前任务实体, 通常包含问题和其它上下文信息（未直接用到, 保留接口规范）。
        execution_history: 前序节点的执行历史结果（所有节点的中间产物）。

    Returns:
        Dict[str,Any]: 
            "SQL" : 多数派融合/投票后最终选定SQL字符串
            "SQL_correct_vote": 按correct_ans字段投票出的SQL（修正后最多、最优）
            "nonecase": 是否所有候选都无可用答案（True/False）
    """
    # 从align_correct节点的输出中获得全部vote轨迹
    vote = get_last_node_result(execution_history, "align_correct")["vote"]
    # 从candidate_generate节点获得SQL原始列表（兜底用）
    SQLs = get_last_node_result(execution_history, "candidate_generate")["SQL"]

    # 优先投票选出纠错后(correct_ans)的最优SQL
    ans_correct, maxm, min_t, vote_M = vote_single(vote, "correct_ans", SQLs)
    # 如需对齐版本投票，可放开下行（通常只需answer/correct_ans即可）
    # align_ans,maxm,min_t,vote_M=vote_single(vote,"align_ans",SQLs)
    # 投票主出口，标准答案answer字段
    ans, maxm, min_t, vote_M = vote_single(vote, "answer", SQLs)
    print(ans)
    print("_______")
    # 判断是否所有候选均未得票（皆无可用答案），用于兜底逻辑处理
    nonecase = maxm == 0
    print(
        f"******votes:{vote_M} max vote: {maxm}, min_t:{min_t}, SQL vote is: {ans}"
    )
  
    response = {
        "SQL": ans,                    # 最终投票定夺SQL
        "SQL_correct_vote": ans_correct,   # 纠错派的多数票SQL
        # "SQL_align_vote": align_ans,  # 如果需要可加，各种风格/对齐投票SQL
        "nonecase": nonecase
    }

    return response

