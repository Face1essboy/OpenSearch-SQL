from typing import Any


import pandas as pd
import re
import torch

class ColumnRetriever:
    """
    列检索器，负责根据输入问题（query），数据库以及全体列集合，检索最相关的数据库列集合作为候选。
    支持语义检索、问句词汇启发、多轮筛选，核心用于pipeline的column相关步骤。
    """
    def __init__(self, bert_model, tables_info_dir):
        """
        初始化列检索器。
        :param bert_model: 句向量模型（如bge-m3），用于列/短语向量化、相似度检索。
        :param tables_info_dir: 数据库表结构信息的json路径，用于加载原始列名和别名。
        """
        self.bert_model = bert_model
        self.tables_info_dir = tables_info_dir

    def get_col_retrieve(self, q, db, l):
        """
        主接口：针对问题q、数据库名db以及所有列全名l（如table.column），返回最相关的列集合。
        - 结合问题内容分词、启发词和语义检索获得主/别名两套候选，最后合并。
        :param q: str，问题文本。
        :param db: str，数据库名称/ID。
        :param l: list[str]，所有table.column格式的列名。
        :return: set[str]，筛选出的相关列名（含表前缀）。
        """

        # 1. 生成问题分词n-gram特征
        ext_a = list[Any](self.get_kgram(q))
        
        # 2. 基于问句wh-word的启发规则召回对应语义列名（如"which"召回"name", "date"等）
        recall_l = set[Any](re.findall("who|which|where|when", q, re.IGNORECASE))
        recall_dic = {
            "who": ["name"],
            "which": ["name", "location", "id", "date", "time"],
            "when": ["time", "date"],
            "where": ["country", "place", "location", "city"]
        }
        for x in recall_l:
            ext_a.extend(recall_dic[x.lower()])  # 将规则召回关键词加入检索短语
        
        # 3. 将所有table.column形式的列转为{"列名": set(所有表.列)}映射
        table_dic = self.get_tab_col_dic(l)  # 所有列名对应的table.column集合（去除表前缀/反引号）
        l = list(table_dic.keys())           # 所有唯一的列名
        
        # 4. 对正式列名（如数据库schema中的）做一次语义匹配检索
        all_col = self.col_ret(l, ext_a)
        
        # 5. 读取列别名信息（如自然语言转换列名）并对别名列做一轮检索增强
        tab_df = pd.read_json(self.tables_info_dir)
        col_name_d = self.col_name_dic(tab_df, db)  # {别名列: 原始列}
        
        re_col = []
        if col_name_d:
            col_l = list(col_name_d.keys())
            re_col = self.col_ret(col_l, ext_a)  # 对别名（如question/别名）做一次语义检索
        
        # 6. 最终结果合并（将别名转回schema真实列名，否则取别名对应的全部表前缀全名）
        ans = self.get_col_set(all_col, re_col, col_name_d, table_dic)
        
        return ans

    def get_kgram(self, q, k=5):
        """
        提取问题文本的1~(k-1)-gram分词短语（连续子串），用于后续和列名的向量语义对齐。
        :param q: str，问题文本。
        :param k: int，最大gram长度（默认为5，能覆盖简单修饰短语）。
        :return: set[str]，所有长度为1~(k-1)的短语集合。
        """
        # 统一分隔符，将标点转空格，规范化空格
        q = re.sub(r'[^\s\w]', ' ', q)
        q = re.sub(r'\s+', ' ', q)
        q_l = q.split(' ')
        s = set()
        # 提取所有长度为i的子连续短语（window滑动）
        for i in range(1, k):
            for j in range(len(q_l) - i + 1):
                s.add(" ".join(q_l[j:i + j]))
        return s

    def get_tab_col_dic(self, table_list):
        """
        将'表.列'形式转为{'列名': set(table.col)}，方便后续只按列名检索，再按表补全所有同名列。
        :param table_list: list[str]，原始形如'table.column'的列全名。
        :return: dict[str, set[str]]，映射关系。
        """
        tab_dic = {}
        for x in table_list:
            t, col = x.split(".")
            col = col.strip('`')  # 去除反引号更通用
            tab_dic.setdefault(col, set())
            tab_dic[col].add(x)
        return tab_dic

    def col_ret(self, l, ext_a):
        """
        对给定候选列集l和问题短语集ext_a，执行BERT语义检索（余弦相似度矩阵）。
        返回最相关的前n个候选列名（主流程所有地方都通过此聚合）。
        :param l: list[str]，候选列列表（如schema列/别名）。
        :param ext_a: list[str]，问题检索短语（n-gram+启发词）。
        :return: set[str]，选中的相关列名。
        """
        # 列名转向量
        l_emb = self.bert_model.encode(
            l,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        num_pick = min(4, len(l))  # 最多保留前4个相关列（可调优）
        # 检索短语转向量并计算匹配分数（问题短语矩阵 * 列向量矩阵T）
        m_ans = self.bert_model.encode(
            ext_a,
            convert_to_tensor=True,
            show_progress_bar=False
        ) @ l_emb.T
        all_col = self.same_pick(l, m_ans, num_pick)
        return all_col

    def col_name_dic(self, df, db):
        """
        读取表结构文件中该db的（别名列，原始列）信息。
        仅保留不相等（即存在别名）部分。
        :param df: DataFrame，全集表结构信息。
        :param db: str，目标数据库名。
        :return: dict[str, str]，{别名列: 原始列}的映射。
        """
        a, b = df[df["db_id"] == db][["column_names", "column_names_original"]].values[0]
        # 跳过第0项（通常为('*', '*')，只取实际列），只保留别名和原始列不一致项
        return {x[1]: y[1] for x, y in zip(a[1:], b[1:]) if x[1] != y[1]}

    def get_col_set(self, all_col, re_col, col_name_d, table_dic, reflect=False):
        """
        合并两类检索结果（主列/别名列），最终返回所有相关列集合（全名形式）。
        若reflect为True，则转回原始schema命名（保留反引号）。
        :param all_col: set[str]，主列名检索结果。
        :param re_col: set[str]，别名列名检索结果。
        :param col_name_d: dict[str,str]，别名到原始列名映射。
        :param table_dic: dict[str, set[str]]，列名逆向查表.
        :param reflect: bool，是否输出规范schema名。
        :return: set[str]，最终全部相关列（带表前缀）。
        """
        ans = set()
        if reflect:
            # 直接输出列名（带反引号格式，常用于SQL语法）
            for x in all_col:
                if x.find(" ") != -1:
                    x = f"`{x}`"
                ans.add(x)
            for x in re_col:
                tmp = col_name_d[x]
                if tmp.find(" ") != -1:
                    tmp = f"`{tmp}`"
                ans.add(tmp)
            return ans
        # 默认操作：用表前缀补全所有同名列/别名映射
        for x in all_col:
            ans = ans.union(table_dic[x])
        for x in re_col:
            real_name = col_name_d[x]
            if real_name not in all_col:
                ans = ans.union(table_dic[real_name])
        return ans

    def same_pick(self, l, m_ans, num_pick, shold=0.7):
        """
        根据BERT相似度得分，选取得分大于shold的top k列，返回列名集合。
        :param l: list[str]，候选列集合。
        :param m_ans: torch.Tensor，问题与列的相似度矩阵。
        :param num_pick: int，最多选取的列数。
        :param shold: float，相似度最小阈值。
        :return: set[str]，最终筛选的高相关列集合。
        """
        # torch.topk按列（每个问题子串）取最大num_pick名次，再筛去低分部分
        all_col = set((
            torch.topk(
                m_ans,
                num_pick
            ).indices[
                torch.topk(m_ans, num_pick).values > shold
            ]).tolist()
        )
        all_col = set(l[x] for x in all_col)
        return all_col

