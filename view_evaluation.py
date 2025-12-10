#!/usr/bin/env python3
"""
查看评估结果的脚本
用法: python view_evaluation.py [result_json_file]
"""
import json
import sys
import os
from pathlib import Path

def view_evaluation_result(json_file):
    """查看单个文件的评估结果"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 查找 evaluation 节点
    evaluation_node = None
    for node in data:
        if node.get("node_type") == "evaluation":
            evaluation_node = node
            break
    
    if not evaluation_node:
        print(f"未找到 evaluation 节点 in {json_file}")
        return
    
    print("=" * 80)
    print(f"评估结果: {Path(json_file).name}")
    print("=" * 80)
    
    # 获取任务信息
    task_info = None
    for node in data:
        if "task" in str(node):
            # 尝试从执行历史中找到任务信息
            pass
    
    # 显示各个节点的评估结果
    nodes_to_evaluate = ["candidate_generate", "align_correct", "vote"]
    
    for node_name in nodes_to_evaluate:
        if node_name not in evaluation_node:
            continue
            
        result = evaluation_node[node_name]
        print(f"\n【{node_name}】")
        print("-" * 80)
        
        # 评估结果
        exec_res = result.get("exec_res", "N/A")
        exec_err = result.get("exec_err", "--")
        
        # 判断是否正确
        if exec_res == 1:
            status = "✅ 正确"
        elif exec_res == 0:
            status = "❌ 错误"
        elif exec_res == "generation error":
            status = "⚠️  生成错误"
        else:
            status = f"❓ 未知状态: {exec_res}"
        
        print(f"状态: {status}")
        print(f"执行结果: {exec_res}")
        if exec_err != "--":
            print(f"错误信息: {exec_err}")
        
        # SQL 对比
        gold_sql = result.get("GOLD_SQL", "N/A")
        predicted_sql = result.get("PREDICTED_SQL", "N/A")
        
        print(f"\n问题: {result.get('Question', 'N/A')}")
        print(f"证据: {result.get('Evidence', 'N/A')}")
        print(f"\n标准答案 SQL:")
        print(f"  {gold_sql}")
        print(f"\n预测 SQL:")
        print(f"  {predicted_sql}")
        
        # 比较两个SQL是否相同
        if gold_sql != "N/A" and predicted_sql != "N/A" and predicted_sql != "--":
            if gold_sql.strip() == predicted_sql.strip():
                print("  ⚠️  注意: SQL字符串完全相同，但执行结果可能不同（语义等价）")
    
    print("\n" + "=" * 80)

def view_statistics(stat_file):
    """查看统计文件"""
    with open(stat_file, 'r') as f:
        stats = json.load(f)
    
    print("=" * 80)
    print("整体统计结果")
    print("=" * 80)
    
    counts = stats.get("counts", {})
    for node_name, node_stats in counts.items():
        print(f"\n【{node_name}】")
        total = node_stats.get("total", 0)
        correct = node_stats.get("correct", 0)
        incorrect = node_stats.get("incorrect", 0)
        error = node_stats.get("error", 0)
        
        print(f"  总数: {total}")
        print(f"  正确: {correct}")
        print(f"  错误: {incorrect}")
        print(f"  生成错误: {error}")
        
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"  准确率: {accuracy:.2f}%")

def main():
    if len(sys.argv) > 1:
        # 指定了文件路径
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            if file_path.endswith("statistics.json") or file_path.endswith("-statistics.json"):
                view_statistics(file_path)
            else:
                view_evaluation_result(file_path)
        else:
            print(f"文件不存在: {file_path}")
    else:
        # 查找最新的结果目录
        results_dir = Path("results")
        if not results_dir.exists():
            print("results 目录不存在")
            return
        
        # 查找最新的 dev 结果
        dev_results = list(results_dir.glob("dev/*/*/*/"))
        if not dev_results:
            print("未找到结果目录")
            return
        
        latest_dir = max(dev_results, key=os.path.getmtime)
        
        # 显示统计信息
        stat_file = latest_dir / "-statistics.json"
        if stat_file.exists():
            view_statistics(stat_file)
        
        # 显示最新的评估结果文件
        json_files = list(latest_dir.glob("*.json"))
        json_files = [f for f in json_files if not f.name.startswith("-")]
        
        if json_files:
            print(f"\n最新的评估结果文件在: {latest_dir}")
            print(f"找到 {len(json_files)} 个结果文件")
            
            # 显示第一个文件作为示例
            if json_files:
                print("\n" + "=" * 80)
                view_evaluation_result(json_files[0])

if __name__ == "__main__":
    main()

