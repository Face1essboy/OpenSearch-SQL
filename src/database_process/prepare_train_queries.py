import argparse, os, sys, re, tqdm
import pandas as pd
import logging

# 添加父路径到sys.path，方便import本地模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.model import model_chose  # 导入模型选择函数

## model "gpt-4 32K" "gpt-3.5-16K-1106"
## add parse to train.json
# 设置logging的基本输出格式和等级
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def convert_table(s, sql):
#     """
#     替换SQL字符串中的表别名为原表名，使下游处理更加规范。
#     Args:
#         s (str): 需要替换的字符串
#         sql (str): 包含表别名的SQL
#     Returns:
#         str: 替换后的字符串
#     """
#     # 匹配 SQL 语句中 "表名 AS 别名"
#     l = re.findall(' ([^ ]*) +AS +([^ ]*)', sql)
#     for li in l:
#         # 用原表名替换掉别名前缀
#         s = s.replace(f" {li[1]}.", f" {li[0]}.")
#     return s

# def parse_ans(sql, ans):
#     """
#     用于解析模型输出，将关键内容组合成指定格式。
#     Args:
#         sql (str): 原始SQL语句
#         ans (str): 模型输出（包含理由、列、值等）
#     Returns:
#         str: 按规范拼接的fewshot模板字符串
#     """
#     # 清理富文本标记
#     ans = ans.replace('```\n', '').replace('```', '')
#     # 将别名替换为原始表名
#     ans = convert_table(ans, sql)
#     # 提取各个部分
#     reason = re.search("#reason:.*", ans).group()
#     column = re.search("#columns:.*", ans).group()
#     values = re.search("#values:.*", ans).group()
#     select = re.search("#SELECT:.*", ans).group()
#     sqllike = "#SQL-Like:" + re.search("#SQL-[Ll]ike:(.*)", ans).groups()[0]
#     # 按预设顺序组合
#     final_str = "\n".join([reason, column, values, select, sqllike, f"#SQL: {sql}"])
#     return final_str

def extract_ans(sql, ans):
    """
    提取reason、column以及SQL中的常量值，格式化为简洁的fewshot样例内容。
    Args:
        sql (str): SQL语句
        ans (str): fewshot原始内容
    Returns:
        str: 按格式拼接的字符串
    """
    reason = re.search("#reason:.*", ans).group()
    column = re.search("#columns:.*", ans).group()
    # 提取SQL中的单引号内容（常量）
    vals = re.findall("'((?:''|[^'])*)'", sql)
    vals_f = [f"\"{x}\"" for x in vals if x != "%Y"]
    final_str = f"{reason}\n{column}\n#values: {', '.join(vals_f)}"
    return final_str

def prepare_train_queries(data_dir, new_train_dir, model, start=0, end=9427):
    """
    从预处理的数据生成训练fewshot样例，并保存，支持分段和多次尝试。
    Args:
        data_dir (str): 数据目录路径
        new_train_dir (str): 输出文件路径
        model: 模型名称 (或对象)
        start (int): 起始样本下标
        end (int): 终止样本下标（不包含）
    Returns:
        None
    """
    # 构建预处理后的train.json路径
    train_json = os.path.join(data_dir, 'data_preprocess', 'train.json')

    # 加载train.json数据为DataFrame
    df = pd.read_json(train_json)

    # 遍历每一行数据，逐条处理
    for i in tqdm.tqdm(range(start, end), total=end - start):
        for _ in range(3):  # 每条样本最多尝试3次
            try:
                # 读取当前行的question、evidence、SQL
                q = df.iloc[i]['question']
                e = df.iloc[i]["evidence"]
                sql = df.iloc[i]["SQL"]
                # 调用大模型生成fewshot格式内容
                content = model_chose("prepare_train_queries", model).fewshot_parse(q, e, sql)
                # 保存完整fewshot内容
                df.loc[i, 'parse'] = content.strip() + "\n#SQL: " + sql
                # 保存这样只提取reason/columns/values的简洁fewshot内容
                df.loc[i, 'extract'] = extract_ans(sql, content)
                break   # 成功则跳出重试循环
            except Exception as e:
                # 捕获异常并打印，继续下一次尝试
                print(f"Error processing row {i}: {str(e)}")

    # 保存处理后的DataFrame到新的json文件
    df[start:end].to_json(new_train_dir, orient='records', indent=4)

if __name__ == '__main__':
    # 配置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_root_directory',
                        type=str,
                        help='data path',
                        default="Bird")
    parser.add_argument('--model',
                        type=str,
                        help='model',
                        default="gpt-4o-mini-0718")
    parser.add_argument('--start',
                        type=int,
                        help='start_point',
                        default=0)
    parser.add_argument('--end',
                        type=int,
                        help='end_point',
                        default=9428)
    args = parser.parse_args()

    # 输出日志，提示保存文件路径
    logging.info(f"Start generate_fewshot_step_1,the output_file is {args.db_root_directory}/llm_train_parse.json")
    llm_train_json = os.path.join(args.db_root_directory, 'llm_train_parse.json')
    prepare_train_queries(args.db_root_directory, llm_train_json, args.model, args.start, args.end)

