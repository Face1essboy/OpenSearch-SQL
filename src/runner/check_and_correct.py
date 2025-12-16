# -- 基本库导入与工具 --
import os
import sqlite3
import re
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor, TimeoutError
import random
import time
from func_timeout import func_timeout, FunctionTimedOut

# =============================
# 辅助工具函数、正则预处理等
# =============================

def sql_raw_parse(sql, return_question):
    """
    解析LLM输出，提取 #SQL: 及前置 question 部分，清理 ```, /* 等多余内容。
    return_question = True 时返回(question, sql)。默认仅返回sql。
    """
    sql = sql.split('/*')[0].strip().replace('```sql', '').replace('```', '')
    sql = re.sub("```.*?", '', sql)
    rwq = None
    if return_question:
        rwq, sql = sql.split('#SQL:')
    else:
        sql = sql.split('#SQL:')[-1]
    if sql.startswith("\"") or sql.startswith("\'"):  # 去除引号包裹
        sql = sql[1:-1]
    sql = re.sub('\s+', ' ', sql).strip()
    return sql, rwq

def get_sql(chat_model,
            prompt,
            temp=1.0,
            return_question=False,
            top_p=None,
            n=1,
            single=True):
    """
    向 chat_model 发送 prompt，获取 SQL 结果。
    若 single==True: 返回(sql, question)，否则返回消息内容列表。
    """
    sql = chat_model.get_ans(prompt, temp, top_p=top_p, n=n, single=single)
    # 检查API调用是否失败情况
    if sql is None:
        if single:
            return None, None
        else:
            return None, None
    if single:
        return sql_raw_parse(sql, return_question)
    else:
        return [x['message']['content'] for x in sql], ""

def retable(sql):
    """
    将SQL中的别名T1.T2等恢复为原表名（消歧），如 Paper AS T1 => Paper.
    便于进一步的正则处理和定位。
    """
    table_as = re.findall(' ([^ ]*) +AS +([^ ]*)', sql)
    for x in table_as:
        sql = sql.replace(f"{x[1]}.", f"{x[0]}.")
    return sql

def max_fun_check(sql_retable):
    """
    检查SQL中的聚合函数(如= (SELECT MAX(...)), SELECT MIN(x)...)等典型歧义和嵌套写法。
    返回：聚合函数、order by子查询、select中的聚合（如select min(x)）三种类型
    """
    fun_amb = re.findall("= *\( *SELECT *(MAX|MIN)\((.*?)\) +FROM +(\w+)", sql_retable)
    order_amb = set(re.findall("= (\(SELECT .* LIMIT \d\))", sql_retable))
    select_amb = set(re.findall("^SELECT[^\(\)]*? ((MIN|MAX)\([^\)]*?\)).*?LIMIT 1", sql_retable))
    return fun_amb, order_amb, select_amb

def foreign_pick(sql):
    """
    提取SQL中JOIN ... ON部分的所有 table.column 形式外键字段
    返回所有出现等式关系的字段集合。
    """
    matchs = re.findall("ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+) ", sql)
    ma_all = [x for y in matchs for x in y]
    return set(ma_all)

def column_pick(sql, db_col, foreign_set):
    """
    检查SQL中是否有潜在的列名歧义（即同名字段出现在多个表）。
    db_col: 形如 table.column 的列表。
    返回可能歧义的列及其可能来自的表
    """
    matchs = foreign_pick(sql)
    cols = set()
    col_table = {}
    ans = set()
    sql_select = set(re.findall("SELECT (.*?) FROM ", sql))
    for x in db_col:  # 统计所有表的同名字段归属
        if sql.find(x) != -1:
            cols.add(x)
        table, col = x.split('.')
        col_table.setdefault(col, [])
        col_table[col].append(table)
    for col in cols:
        table, col_name = col.split('.')
        flag = True
        for x in sql_select:
            if x.find(col) != -1:
                flag = False
                break
        # 如果既在外键又未被select，跳过
        if flag and (col in foreign_set or x in matchs):
            continue
        if col_table.get(col_name):
            ambiguity = []
            for t in col_table[col_name]:
                tbc = f"{t}.{col_name}"
                if tbc != col:
                    ambiguity.append(tbc)
            if len(ambiguity):
                amb_des = col + ": " + ", ".join(ambiguity)
                ans.add(amb_des)
    return sorted(list(ans))

def values_pick(vals, sql):
    """
    检查SQL中的where条件是否使用了values表中的值，但出现在了错误的列或者没出现等异常。
    返回建议更正的字段描述。
    """
    val_dic = {}
    ans = set()
    try:
        for val in vals:
            val_dic.setdefault(val[1], [])
            val_dic[val[1]].append(val[0])
        for val in val_dic:
            in_sql, not_sql = [], []
            if sql.find(val):
                for x in val_dic[val]:
                    if sql.find(x) != -1:
                        in_sql.append(f"{x} = '{val}'")
                    else:
                        not_sql.append(f"{x} = '{val}'")
            if len(in_sql) and len(not_sql):
                ans.add(f"{', '.join(in_sql)}: {', '.join(not_sql)}")
        return sorted(list(ans))
    except:
        return []

def func_find(sql):
    """
    查找SQL嵌套聚合函数模式，并返回如何重写的建议
    """
    fun_amb = re.findall("\( *SELECT *(MAX|MIN)\((.*?)\) +FROM +(\w+)", sql)
    fun_str = []
    for fun in fun_amb:
        fuc = fun[0]
        col = fun[1]
        table = fun[2]
        if fuc == "MAX":
            order = "DESC"
        else:
            order = "ASC"
        str_fun = f"(SELECT {fuc}({col}) FROM {table}): ORDER BY {table}.{col} {order} LIMIT 1"
        fun_str.append(str_fun)
    return "\n".join(fun_str)

# 正则辅助
t1_tabe_value = re.compile("(\w+\.[\w]+) =\s*'([^']+(?:''[^']*)*)'")          # table.col = 'value'
t2_tab_val   = re.compile("(\w+\.`[^`]*?`) =\s*'([^']+(?:''[^']*)*)'")        # table.`col` = 'value'

# ==============================
# SQL替换和JOIN纠正工具
# ==============================

def filter_sql(b, bx, conn, SQL, chars=""):
    """
    用于修正JOIN...IN或OR条件中SQL，依次尝试分解条件的替换，返回能成功执行的SQL及是否修正标志。
    b: 条件分量/选项。bx: 原始包含IN/OR的字符串
    chars: 用于替换IN为=。conn为数据连接。
    """
    flag = False
    for x in b:
        sql_t = SQL.replace(bx, f"{chars}{x}")
        try:
            df = pd.read_sql_query(sql_t, conn)
        except Exception as e:
            print(e)
            df = []
        # 有返回即有效更正
        if len(df):
            SQL = sql_t
            flag = True
            break
    return SQL, flag

def join_exec(db, bx, al, question, SQL, chat_model):
    """
    尝试从“JOIN ... IN/OR ...”的模糊条件中，挑选能正常执行的JOIN条件。
    若未能修正交给gpt自动处理。
    """
    flag = False
    with sqlite3.connect(db, timeout=180) as conn:
        if bx.startswith("IN"):
            b = bx[2:].strip(" ()").split(',')
            SQL, flag = filter_sql(b, bx, conn, SQL, chars="= ")
        elif al.find("OR") != -1:
            a = al.split("OR")
            SQL, flag = filter_sql(a, al, conn, SQL)
    return SQL, flag

def gpt_join_corect(SQL, question, chat_model):
    """
    利用大模型补全JOIN条件的裁剪，让JOIN ON条件只保留最高优先级的=判断（去除冗余OR,IN）。
    """
    prompt = f"""下面的question对应的SQL错误的使用了JOIN函数,使用了JOIN table AS T ON Ta.column1 = Tb.column2 OR Ta.column1 = Tb.column3或JOIN table AS T ON Ta.column1 IN的JOIN方式,请你只保留 OR之中优先级最高的一组 Ta.column = Tb.column即可.

question:{question}
SQL: {SQL}

请直接给出新的SQL, 不要回复任何其他内容:
#SQL:"""
    SQL = get_sql(chat_model, prompt, 0.0)[0].split("SQL:")[-1]
    return SQL

def select_check(SQL, db_col, chat_model, question):
    """
    强制将SQL中select concat连接符 '|| ' ' ||' 替换为逗号分隔。
    如果有模糊select *写法，用prompt要求重写为明确列select id。
    """
    select = re.findall("^SELECT.*?\|\| ' ' \|\| .*?FROM", SQL)
    if select:
        SQL = SQL.replace("|| ' ' ||", ', ')
    select_amb = re.findall("^SELECT.*? (\w+\.\*).*?FROM", SQL)
    if select_amb:
        prompt = f"""数据库存在以下字段:
{db_col}
现有问题为 {question}
SQL:{SQL}
我们规定视这种不明确的查询为对应的id
现在请你把上面SQL的{select_amb[0]}改为对应的id,请你直接给出SQL, 不要回复任何其他内容:
#SQL:"""
        SQL = get_sql(chat_model, prompt, 0.0)[0].split("SQL:")[-1]
    return SQL

# ===============================
# SQL校正与纠错核心 soft_check 类
# ===============================

class soft_check:
    """
    SQL纠错逻辑主类，封装对SQL的风格、函数、JOIN、值等自动化修正的策略。
    依赖于BERT及chat_model等注入的对象。
    """

    def __init__(self,
                 bert_model,
                 chat_model,
                 soft_prompt,
                 correct_dic,
                 correct_prompt,
                 vote_prompt="") -> None:
        self.bert_model = bert_model
        self.chat_model = chat_model
        self.soft_prompt = soft_prompt
        self.correct_dic = correct_dic
        self.correct_prompt = correct_prompt
        self.vote_prompt = vote_prompt

    def vote_chose(self, SQLs, question):
        """
        投票选择最优SQL（自一致性），将所有SQL合并prompt后用大语言模型投票。
        """
        all_sql = '\n\n'.join(SQLs)
        prompt = self.vote_prompt.format(question=question, sql=all_sql)
        SQL_vote = get_sql(self.chat_model, prompt, 0.0)[0]
        return SQL_vote

    def soft_correct(self, SQL, question, new_prompt, hint=""):
        """
        使用 soft_prompt prompt 让模型判定SQL是否需要修改，若不对，则激活拯救逻辑或LLM重写。
        """
        soft_p = self.soft_prompt.format(SQL=SQL, question=question, hint=hint)
        soft_SQL = self.chat_model.get_ans(soft_p, 0.0)
        soft_SQL = re.sub("```\w*", "", soft_SQL)
        soft_json = json.loads(soft_SQL)
        if (soft_json["Judgment"] == False or soft_json["Judgment"] == 'False') and soft_json["SQL"] != "":
            SQL = soft_json["SQL"]
            SQL = re.sub('\s+', ' ', SQL).strip()
        elif (soft_json["Judgment"] == False or soft_json["Judgment"] == 'False'):
            SQL = get_sql(self.chat_model, new_prompt, 1.0, False)[0]
        return SQL, soft_json["Judgment"]

    def double_check(
            self,
            new_prompt,
            values: list,
            values_final,
            SQL: str,
            question: str,
            new_db_info: str,
            db_col: list,
            db: str,  # db路径
            hint="") -> str:
        """
        标准对齐修正流程，依次进行值校正、JOIN纠错、函数替换、时间格式、NULL条件等逐级修正。
        """
        SQL = re.sub("(COUNT)(\([^\(\)]*? THEN 1 ELSE 0.*?\))", r"SUM\2", SQL)
        sql_retable = retable(SQL)
        SQL = self.values_check(sql_retable, values, values_final, SQL, question, new_db_info, db_col, hint)
        SQL = self.JOIN_error(SQL, question, db)
        SQL = self.func_check(sql_retable, SQL, question)
        SQL = self.func_check2(question, SQL)  # ORDER BY (MIN|MAX).* LIMIT
        SQL = self.time_check(SQL)
        SQL = self.is_not_null(SQL)
        SQL = select_check(SQL, db_col, self.chat_model, question)
        return SQL, True

    def double_check_style_align(
        self,
        SQL: str,
        question: str,
        db_col: list,
        sql_retable: str,
    ) -> str:
        """
        风格一致性对齐，仅做函数正则、is not null、select校正等（不改数值）。
        """
        SQL = self.func_check(sql_retable, SQL, question)
        SQL = self.is_not_null(SQL)
        SQL = select_check(SQL, db_col, self.chat_model, question)
        return SQL, True

    def double_check_function_align(
        self,
        SQL: str,
        question: str,
        db: str,  # db路径
    ) -> str:
        """
        仅对SQL中函数相关的典型问题进行修正，包括JOIN与时间表达式。
        """
        SQL = self.JOIN_error(SQL, question, db)
        SQL = self.func_check2(question, SQL)  # ORDER BY (MIN|MAX).* LIMIT
        SQL = self.time_check(SQL)
        return SQL, True

    def double_check_agent_align(
        self,
        sql_retable: str,
        values: list,
        values_final,
        SQL: str,
        question: str,
        new_db_info: str,
        db_col: list,
        hint=""
    ) -> str:
        """
        仅做值级别修正（即where条件校对），不涉及join/func风格/格式修复等。
        """
        SQL = self.values_check(sql_retable, values, values_final, SQL, question, new_db_info, db_col, hint)
        return SQL, True

    def JOIN_error(self, SQL, question, db):
        """
        修正SQL中JOIN的多等式、IN等冗余情况，对无法修正的交给大模型二次纠正。
        """
        join_mutil = re.findall(
            "JOIN\s+\w+(\s+AS\s+\w+){0,1}\s+ON(\s+\w+\.\w+\s*(=\s*\w+\.\w+(?:\s+OR\s+\w+\.\w+\s*=\s*\w+\.\w+)+|IN\s+\(.*?\)))",
            SQL)
        flag = False
        if join_mutil:
            _, al, bx = join_mutil[0]
            try:
                SQL, flag = func_timeout(180 * 8, join_exec, args=(db, bx, al, question, SQL, self.chat_model))
            except FunctionTimedOut:
                print("time out join")
            except Exception as e:
                print(e)
        if not flag and join_mutil:  # 没有修正成功直接gpt二次修正
            SQL = gpt_join_corect(SQL, question, self.chat_model)
            print("soft change JOIN gpt")
        return SQL

    def is_not_null(self, SQL):
        """
        若SQL为ORDER BY ... LIMIT ...，且没有判断SUM/COUNT情况，则增加WHERE IS NOT NULL安全条件。
        """
        SQL = SQL.strip()
        inn = re.findall("ORDER BY .*?(?<!DESC )LIMIT +\d+;{0,1}", SQL)
        if not inn:
            return SQL
        for x in inn:
            if re.findall("SUM\(|COUNT\(", x):
                return SQL
        prompt = f"""请你为下面SQL ORDER BY的条件加上WHERE IS NOT NULL限制:
SQL:{SQL}

请直接给出新的SQL, 不要回复任何其他内容:
#SQL:"""
        SQL = get_sql(self.chat_model, prompt, 0.0)[0].split("SQL:")[-1]
        return SQL

    def time_check(self, sql):
        """
        如果SQL语句含有 strftime(...) >= 2020 等年份，强制加上引号
        """
        time_error_fix = re.sub("(strftime *\([^\(]*?\) *[>=<]+ *)(\d{4,})",
                                r"\1'\2'", sql)
        return time_error_fix

    def func_check2(self, question, SQL):
        """
        修正 SQL 中 order by min/max(...) limit 等函数嵌套风格错误
        """
        res = re.search("ORDER BY ((MIN|MAX)\((.*?)\)).*? LIMIT \d+", SQL)
        if res:
            prompt = f"""对于下面的qustion和SQL:
#question: {question}
#SQL: {SQL}

ERROR: {res.group()} 是一种不正确的用法, 请对SQL进行修正, 注意如果SQL中存在GROUP BY, 请判断{res.groups()[0]}的内容是否需要使用 SUM({res.groups()[2]})

请直接给出新的SQL, 不要回复任何其他内容:"""
            SQL = get_sql(self.chat_model, prompt, 0.1)[0]
        return SQL

    def func_check(self, sql_retable, sql, question):
        """
        发现并要求重写所有MAX/MIN嵌套select/group by以及select子句中的聚合函数方式为标准join/order by形式
        """
        fun_amb, order_amb, select_amb = max_fun_check(sql_retable)
        if not fun_amb and not order_amb and not select_amb:
            return sql
        fun_str = []
        origin_f = []
        for fun in fun_amb:
            fuc = fun[0]
            col = fun[1]
            table = fun[2]
            if fuc == "MAX":
                order = "DESC"
            else:
                order = "ASC"
            str_fun = f"WHERE {col} = (SELECT {fuc}({col}) FROM {table}): 请用 ORDER BY {table}.{col} {order} LIMIT 1 代替嵌套SQL"
            origin_f.append(f"WHERE {col} = (SELECT {fuc}({col}) FROM {table})")
            fun_str.append(str_fun)
        for fun in order_amb:
            origin_f.append(fun)
            fun_str.append(f"{fun}: 使用JOIN 形式代替嵌套")
        for fun in select_amb:
            origin_f.append(fun[0])
            fun_str.append(f"{fun[0]}: {fun[1]} function 函数 冗余,请更改")

        func_amb = "\n".join(fun_str)
        prompt = f"""对于下面的问题和SQL, 请根据ERROR和#change ambuity修改:
#question: {question}
#SQL: {sql}
ERROR:{",".join(origin_f)} 不符合要求, 请使用 JOIN ORDER BY LIMIT 形式代替
#change ambuity: {func_amb}

请直接给出新的SQL, 不要回复任何其他内容:"""
        sql = get_sql(self.chat_model, prompt, 0.0)[0]
        return sql

    def values_check(self,
                     sql_retable,
                     values,
                     values_final,
                     sql,
                     question,
                     new_db_info,
                     db_col,
                     hint=""):
        """
        检查SQL的where子句使用的值是否与数据库实际value映射合理，不合理则修正提示大模型补全值修正建议。
        """
        dic_v = {}  # 值到列
        dic_c = {}  # 列到值
        l_v = list(set([x[1] for x in values]))  # 所有可用值
        tables = "( " + " | ".join(set([x.split(".")[0] for x in db_col])) + " )"
        for x in values:
            dic_v.setdefault(x[1], [])
            dic_v[x[1]].append(x[0])
            dic_c.setdefault(x[0], [])
            dic_c[x[0]].append(x[1])
        value_sql = re.findall(t1_tabe_value, sql_retable)  # 标准 table.col
        value_sql.extend(re.findall(t2_tab_val, sql_retable))
        tabs = set(re.findall(tables, sql))
        # 单表select x='y'且没表名前缀时转完整
        if len(tabs) == 1:
            val_single = re.findall("[ \(]([\w]+) =\s*'([^']+(?:''[^']*)*)'", sql)
            val_single.extend(re.findall("[ \(]([\w]+) =\s*'([^']+(?:''[^']*)*)'", sql))
            val_single = set(val_single)
            tab = tabs.pop()[1:-1]
            for x in val_single:
                value_sql.append((f"{tab}.{x[0]}", x[1]))
        badval_l = []
        change_val = []
        value_sql = set(value_sql)
        # 依次检查每组值是否出现在正确列
        for tab_val in value_sql:
            tab, val = tab_val
            if len(re.findall("\d", val)) / len(val) > 0.6:
                continue
            tmp_col = dic_v.get(val)
            if not tmp_col and len(l_v):  # 未知值，尝试BERT相似度找最像的修正
                val_close = self.bert_model.encode(val, show_progress_bar=False) @ self.bert_model.encode(
                    l_v, show_progress_bar=False).T
                if val_close.max() > 0.95:
                    val_new = l_v[val_close.argmax()]
                    sql = sql.replace(f"'{val}'", f"'{val_new}'")
                    val = val_new
            tmp_col = dic_v.get(val)
            tmp_val = dic_c.get(tab, {})
            # 若该值虽然存在但不是本table/col的，应全列提示用户
            if tmp_col and tab not in tmp_col:
                lt = [f"{x} ='{val}'" for x in tmp_col]
                lt.extend([f"{x} ='{val}'" for x in tmp_val])
                rep = ", ".join(lt)
                badval_l.append(f"{tab} = '{val}'")
                change_val.append(f"{tab} = '{val}': {rep}")
        # 收集错误后产生prompt给大模型重写SQL
        if badval_l:
            v_l = "\n".join(change_val)
            prompt = f"""Database Schema:
{new_db_info}

#question: {question}
#SQL: {sql}
ERROR: 数据库中不存在: {', '.join(badval_l)}
请用以下条件重写SQL:\n{v_l}

请直接给出新的SQL,不要回复任何其他内容:
#SQL:"""
            sql = get_sql(self.chat_model, prompt, 0.0)[0]
        return sql

    # ========================================
    # SQL最终执行校验与死循环自动重生机制
    # ========================================
    def correct_sql(self,
                   db_sqlite_path,
                   sql,
                   query,
                   db_info,
                   hint,
                   key_col_des,
                   new_prompt,
                   db_col={},
                   foreign_set={},
                   L_values=[]):
        """
        SQL试执行与多次自动校正：如出错则用correct_prompt或new_prompt逐次递增修正SQL
        最多三轮，彻底失败会标记 none_case
        """
        conn = sqlite3.connect(db_sqlite_path, timeout=180)
        count = 0
        raw = sql
        none_case = False
        while count <= 3:
            try:
                df = pd.read_sql_query(sql, conn)
                if len(df) == 0:
                    raise ValueError("Error':Result: None")
                else:
                    break
            except Exception as e:
                if count >= 3:  # 达到最大轮次仍出错，regenerate
                    wsql = sql
                    sql = get_sql(self.chat_model, new_prompt, 0.2)[0]
                    none_case = True
                    break
                count += 1
                tag = str(e)
                e_s = str(e).split("':")[-1]
                result_info = f"{sql}\nError: {e_s}"
            # 如果SQL染病且没SELECT，直接new prompt重生
            if sql.find("SELECT") == -1:
                sql = get_sql(self.chat_model, new_prompt, 0.3)[0]
            else:
                fewshot = self.correct_dic["default"]
                advice = ""
                for x in self.correct_dic:
                    if tag.find(x) != -1:  # 匹配到具体错误原因
                        fewshot = self.correct_dic[x]
                        if e_s == "Result: None":
                            sql_re = retable(sql)
                            adv = column_pick(sql_re, db_col, foreign_set)
                            adv = '\n'.join(adv)
                            val_advs = values_pick(L_values, sql_re)
                            val_advs = '\n'.join(val_advs)
                            func_call = func_find(sql)
                            if len(adv) or len(val_advs) or len(func_call):
                                advice = "#Change Ambiguity: " + "(replace or add)\n"
                                l = [x for x in [adv, val_advs, func_call] if len(x)]
                                advice += "\n\n".join(l)
                        elif x == "no such column":
                            advice += "Please check if this column exists in other tables"
                        break
                fewshot = ""
                advice = ""
                cor_prompt = self.correct_prompt.format(
                    fewshot=fewshot,
                    db_info=db_info,
                    key_col_des=key_col_des,
                    q=query,
                    hint=hint,
                    result_info=result_info,
                    advice=advice)
                sql = get_sql(self.chat_model, cor_prompt, 0.2 + count / 5, top_p=0.3)[0]
            raw = sql
        conn.close()
        return sql, none_case

# ===================
# SQL执行工具
# ===================
def sql_exec(SQL, db):
    """
    直接执行一条SQL，获得所有value集合及耗时。
    """
    with sqlite3.connect(db) as conn:
        s = time.time()
        df = pd.read_sql_query(SQL, conn)
        ans = set(tuple(x) for x in df.values)
        time_cost = time.time() - s
    return ans, time_cost

def get_sql_ans(SQL, db_sqlite_path):
    """
    对SQL套壳超时处理，确保bad sql不会拖慢主流程。
    """
    try:
        ans, time_cost = func_timeout(180, sql_exec, args=(SQL, db_sqlite_path))
    except FunctionTimedOut:
        ans, time_cost = [], 100000
        print("time out")
    except Exception as e:
        ans, time_cost = [], 100000
        print(f"SQL execution error: {e}")
    return ans, time_cost

# ===================
# 拆分对齐流程入口API
# ===================
def process_sql(Dcheck, SQL, L_values, values, question,
                new_db_info, db_col_keys, hint, key_col_des, tmp_prompt, db_col, foreign_set, align_methods, db_sqlite_path):
    """
    分层执行agent_align/style_align/function_align三类SQL对齐校正，每层调不同Dcheck子函数。
    align_methods形式如"agent_align+style_align+function_align"
    SQL经过多次修正，最后返回完整执行历史与答案、主表SQL等所有候选对照。
    """
    node_names = align_methods.split('+')
    align_functions = {
        "agent_align": Dcheck.double_check_agent_align,
        "style_align": Dcheck.double_check_style_align,
        "function_align": Dcheck.double_check_function_align
    }
    SQL = re.sub("(COUNT)(\([^\(\)]*? THEN 1 ELSE 0.*?\))", r"SUM\2", SQL)
    sql_retable = retable(SQL)
    judgment = None
    sql_history = {}
    SQL_correct = SQL
    # 分步走所有对齐修正模块
    for node_name in node_names:
        if node_name in align_functions:
            if node_name == "agent_align":
                SQL, judgment = align_functions[node_name](sql_retable, L_values, values, SQL,
                                                           question, new_db_info, db_col_keys, hint)
            elif node_name == "style_align":
                SQL, judgment = align_functions[node_name](SQL, question, db_col_keys, sql_retable)
            elif node_name == "function_align":
                SQL, judgment = align_functions[node_name](SQL, question, db_sqlite_path)
            sql_history[node_name] = SQL
    align_SQL = SQL
    can_ex = True
    nocse = True
    ans = set()
    time_cost = 10000000
    # 最后交给Dcheck.correct_sql对齐完成的SQL进行最终修正与执行
    try:
        SQL, nocse = func_timeout(540,
                                  Dcheck.correct_sql,
                                  args=(db_sqlite_path, SQL, question, new_db_info,
                                        hint, key_col_des, tmp_prompt, db_col,
                                        foreign_set, L_values))
    except:
        print("timeout")
        can_ex = False
    if can_ex:
        ans, time_cost = get_sql_ans(SQL, db_sqlite_path)
        # align_ans=get_sql_ans(align_SQL, db_sqlite_path)
        # correct_ans=get_sql_ans(SQL_correct, db_sqlite_path)
        align_ans = None
        correct_ans = None
    return sql_history, SQL, ans, nocse, time_cost, align_SQL, align_ans, SQL_correct, correct_ans

def muti_process_sql(Dcheck, SQLs, L_values, values, question,
                     new_db_info, hint, key_col_des, tmp_prompt, db_col, foreign_set, align_methods, db_sqlite_path, n):
    """
    多候选SQL并行比选，对每个SQL并发调用process_sql，对齐、执行并统计所有结果（vote策略）。
    SQLs: {sql->计数}字典。n: 并发数
    返回投票vote各SQL结构与none_case失效标志
    """
    vote = []
    none_case = False
    db_col_keys = db_col.keys()
    # 用ThreadPoolExecutor并发执行每个SQL的全流程处理
    with ThreadPoolExecutor(max_workers=n) as executor:
        future_to_sql = {
            executor.submit(process_sql, Dcheck, SQL, L_values, values, question, new_db_info, db_col_keys, hint,
                            key_col_des, tmp_prompt, db_col, foreign_set, align_methods, db_sqlite_path):
            (SQLs[SQL], SQL)
            for SQL in SQLs
        }
        time_cost = 10000000
        for future in as_completed(future_to_sql):
            count, tmp_SQL = future_to_sql[future]
            try:
                sql_history, SQL, ans, none_c, time_cost, align_SQL, align_ans, SQL_correct, correct_ans = future.result(timeout=700)
                vote.append({
                    "sql_history": sql_history,
                    "sql": SQL,
                    "answer": ans,
                    "count": count,
                    "time_cost": time_cost,
                    "align_sql": align_SQL,
                    "align_ans": align_ans,
                    "correct_sql": SQL_correct,
                    "correct_ans": correct_ans
                })
                none_case = none_case or none_c
            except FunctionTimedOut:
                print(f"Error: Processing SQL timeout for SQL count {count}")
                vote.append({
                    "sql_history": tmp_SQL,
                    "sql": tmp_SQL,
                    "answer": [],
                    "count": 1,
                    "time_cost": time_cost,
                    "align_sql": tmp_SQL,
                    "correct_sql": tmp_SQL,
                    "align_ans": [],
                    "correct_ans": [],
                })
                none_case = True
            except Exception as e:
                print(f"Error processing SQL: {e}")
                vote.append({
                    "sql_history": tmp_SQL,
                    "sql": tmp_SQL,
                    "answer": [],
                    "count": 1,
                    "time_cost": time_cost,
                    "align_sql": tmp_SQL,
                    "correct_sql": tmp_SQL,
                    "align_ans": [],
                    "correct_ans": [],
                })
                none_case = True
    return vote, none_case
