import pandas as pd
import re, sqlite3, os, chardet

# 给定注释：查找 MySQL 格式的外键关系
def find_foreign_keys_MYSQL_like(DATASET_JSON, db_name):
    """
    查找 MySQL 格式的外键关系。

    参数:
        DATASET_JSON (str): 包含数据库架构信息的 JSON 文件路径。
        db_name (str): 目标数据库名。

    返回:
        tuple: 
            - output (str): 外键关系描述串，格式如 "tableA.col1 = tableB.col2, ... "
            - col_set (set): 涉及的所有外键列，全表名（如 "tableA.col1"）的集合
    详细过程:
        1. 读取 JSON 架构信息，转换为 pandas DataFrame。
        2. 移除 'column_names' 和 'table_names' 字段（冗余信息）。
        3. 遍历每一条数据库架构记录，收集所有外键关系，将每条外键关系转换为 [db_id, 源表, 目标表, 源表外键列, 目标表外键列] 格式。
        4. 组织成 DataFrame，便于过滤和后续结构化。
        5. 仅保留符合 db_name 的外键关系。
        6. 将所有外键关系生成为 "tableA.colA = tableB.colB" 格式的字符串，并用英文逗号连接。
        7. 同时收集涉及的所有 "表名.列名"，形成集合 col_set。
    """
    # 步骤1：读取JSON为DataFrame
    schema_df = pd.read_json(DATASET_JSON)

    # 步骤2：移除无关列
    schema_df = schema_df.drop(['column_names', 'table_names'], axis=1)

    f_keys = []
    # 步骤3：收集全部外键关系
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']           # 数据库中所有原始表名
        col_names = row['column_names_original']       # 所有[表索引, 字段名]配对
        foreign_keys = row['foreign_keys']             # 所有外键关系，以成对索引表示

        # 对每个外键关系（由源、目标列的索引组成）
        for foreign_key in foreign_keys:
            first, second = foreign_key               # 外键对应的两个字段（均为索引）
            first_index, first_column = col_names[first]    # 获取源字段的（表索引，字段名）
            second_index, second_column = col_names[second] # 获取目标字段的（表索引，字段名）

            # 记录：[db_id, 源表, 目标表, 源表的外键字段, 目标表的外键字段]
            f_keys.append([
                row['db_id'],
                tables[first_index],
                tables[second_index],
                first_column,
                second_column
            ])
    # 步骤4：整理为结构化DataFrame，便于过滤
    spider_foreign = pd.DataFrame(f_keys, columns=[
        'Database name', 'First Table Name',
        'Second Table Name', 'First Table Foreign Key',
        'Second Table Foreign Key'
    ])

    # 步骤5：仅保留当前db_name的数据
    df = spider_foreign[spider_foreign['Database name'] == db_name]

    # 步骤6/7：生成为“表.字段 = 表.字段”格式串，并收集所有涉及的表-字段名
    output = []
    col_set = set()
    for index, row in df.iterrows():
        # 拼接外键等号关系
        cond_str = f"{row['First Table Name']}.{row['First Table Foreign Key']} = " \
                   f"{row['Second Table Name']}.{row['Second Table Foreign Key']}"
        output.append(cond_str)
        # 收集涉及的“表.列”
        col_set.add(f"{row['First Table Name']}.{row['First Table Foreign Key']}")
        col_set.add(f"{row['Second Table Name']}.{row['Second Table Foreign Key']}")
    output = ", ".join(output)
    return output, col_set

# 给定注释：对字段名进行引用，如果包含特殊字符则加反引号
def quote_field(field_name):
    if re.search(r'\W', field_name):
        return f"`{field_name}`"
    else:
        return field_name

# 给定注释：数据库代理类，用于收集数据库的结构信息、模式等
class db_agent:
    def __init__(self, chat_model) -> None:
        self.chat_model = chat_model

    # 给定注释：获取全部数据库相关信息并汇总
    def get_allinfo(self, db_json_dir, db, sqllite_dir, db_dir, tables_info_dir, model):
        db_info, db_col = self.get_db_des(sqllite_dir, db_dir, model)
        foreign_keys = find_foreign_keys_MYSQL_like(tables_info_dir, db)[0]
        all_info = f"Database Management System: SQLite\n#Database name: {db}\n{db_info}\n#Forigen keys:\n{foreign_keys}\n"
        prompt = self.db_conclusion(all_info)
        db_all = self.chat_model.get_ans(prompt)
        all_info = f"{all_info}\n{db_all}\n"
        return all_info, db_col

    # 给定注释：获取指定表的所有详细信息，包括示例值等
    def get_complete_table_info(self, conn, table_name, table_df):
        """
        获取指定表的所有详细信息，包括示例值等。

        参数:
            conn: sqlite3.Connection 对象，数据库连接。
            table_name: str，表名。
            table_df: pd.DataFrame，表结构文件内容（如csv中解析的），用于获取字段描述和示例值描述。

        返回:
            schema_str: str，该表结构与样例的详细描述（文本化）。
            columns: dict，键为 "table_name.column_name" 的字段，值为包含字段属性元组。
        """
        # 1. 获取当前表的字段基础属性（PRAGMA）
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        columns_info = cursor.fetchall()  # [(cid, name, type, notnull, dflt_value, pk), ...]
        
        # 2. 读取当前表的全部数据（可用于分析字段值）
        df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
        
        # 3. 统计字段是否包含 NULL 及是否唯一（存在重复值）
        contains_null = {column: df[column].isnull().any() for column in df.columns}
        contains_duplicates = {column: df[column].duplicated().any() for column in df.columns}

        # 4. 收集每个字段的注释（字段描述）、示例值描述。主要来源于 table_df 入参
        dic = {}  # 字段名: (字段描述, 示例值描述)
        for _, row in table_df.iterrows():
            try:
                col_description, val_description = "", ""
                col = str(row.iloc[0]).strip()  # 假设第0列为字段名

                # 解析字段描述（如存在第三列/空字符串则忽略）
                if pd.notna(row.iloc[2]):
                    col_description = re.sub(r'\s+', ' ', str(row.iloc[2]))
                if col_description.strip() == col or col_description.strip() == "":
                    col_description = ''

                # 解析示例值描述（如存在第五列/空字符串/与字段名、列描述相同则视为无描述）
                if pd.notna(row.iloc[4]):
                    val_description = re.sub(r'\s+', ' ', str(row.iloc[4]))
                if (val_description.strip() == "" or val_description.strip() == col 
                    or val_description == col_description):
                    val_description = ""

                # 限制描述字符数防止过长
                col_description = col_description[:200]
                val_description = val_description[:200]
                dic[col] = col_description, val_description
            except Exception as e:
                print(e)
                dic[col] = "", ""  # 若异常以空描述赋值

        # 5. 获取首行示例值（如表存在数据则取出一行，用于后续组装 schema）
        try:
            row = list(cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 1").fetchall()[0])
        except Exception as e:
            # 若表为空，则row设为空列表
            row = [None] * len(df.columns)

        # 6. 针对每个字段，用实际值生成最多3个唯一示例值（类型智能转换、如字段全空则保留None）
        for i, col in enumerate(df.columns):
            try:
                vals = df[col].dropna().drop_duplicates().iloc[:3].values  # 最多3个示例
                val_p = []
                for val in vals:
                    try:
                        val_p.append(int(val))
                    except:
                        val_p.append(val)
                if len(vals) == 0:
                    raise ValueError
                row[i] = val_p  # 用3个示例值列表替换原值
            except:
                # 若某列全空/异常 保持现有
                pass

        # 7. 构建 schema 表头及结构明细字符串，用于下游 LLM/工程理解
        schema_str = f"## Table {table_name}:\nColumn| Column Description| Value Description| Type| 3 Example Value\n"
        columns = {}  # 用于返回详细字段索引字典
        for column, val in zip(columns_info, row):
            # PRAGMA table_info输出：[cid, name, type, notnull, dflt_value, pk]
            column_name, column_type, not_null, default_value, pk = column[1:6]
            tmp_col = column_name.strip()
            quoted_col = quote_field(column_name)  # 字段名必要时加反引号
            schema_str += f"{quoted_col}| "

            # 字段描述/样例值描述
            col_des, val_des = dic.get(tmp_col, ["", ""])
            schema_str += f"{col_des}|{val_des}|"

            # 追加类型
            schema_str += f"{column_type}| "

            # 可否为NULL+唯一性（用于生成友好的结构信息）
            include_null = f"{'Include Null' if contains_null.get(tmp_col, True) else 'Non-Null'}"
            unique = f"{'Non-Unique' if contains_duplicates.get(tmp_col, False) else 'Unique'}"

            # 若示例过长，不显示原文
            if len(str(val)) > 360:
                val = "<Long text not displayed>"

            # 汇总至columns结构：每个字段一个详细元组
            columns[f"{table_name}.{quoted_col}"] = (col_des, val_des,
                                                     column_type,
                                                     include_null, unique,
                                                     str(val))
            # 拼装到schema字符串
            schema_str += f"{val}\n"
        
        return schema_str, columns

    # 给定注释：获得数据库结构描述及列信息
    def get_db_des(self, sqllite_dir, db_dir, model):
        """
        获取数据库结构的详细描述和所有字段的详细字典（含注释、字段类型、示例值等），
        主要流程如下：
            1. 打开sqlite数据库连接
            2. 获取数据库各表名（过滤系统表，如sqlite_sequence）
            3. 遍历每个表，根据embedding找表的描述csv（优选相似度较高的）
            4. 读取表的描述csv（如有读取异常则用空DataFrame），提取表字段注释与示例
            5. 调用get_complete_table_info生成表结构字符串与字段详细信息
            6. 累加每个表的schema字符串和所有表字段到总db_info, db_col
            7. 返回所有表结构串接（str）及所有表字段字典

        参数:
            sqllite_dir: sqlite数据库文件路径
            db_dir: 数据库主目录，需包含database_description文件夹
            model: 用于文本embedding的模型实例

        返回:
            db_info: str，所有表的结构详细描述字符串（供LLM理解）
            db_col: dict，所有表字段详细（供后续选择与分析）
        """
        conn = sqlite3.connect(sqllite_dir)  # 连接数据库
        table_dir = os.path.join(db_dir, 'database_description')  # 表结构注释csv存储目录
        sql = "SELECT name FROM sqlite_master WHERE type='table';"
        cursor = conn.cursor()
        tables = cursor.execute(sql).fetchall()   # 获取全部表名
        db_info = []  # 用于累计所有表的schema字符串
        db_col = dict()  # 汇总所有表字段详细信息

        file_list = os.listdir(table_dir)  # 获取所有描述文件名
        files_emb = model.encode(file_list, show_progress_bar=False)  # 对所有描述文件名做embedding
        for table in tables:
            if table[0] == 'sqlite_sequence':
                # 跳过sqlite系统自增序列表
                continue
            # 计算待查表文件名(table.csv)的embedding
            target_file_emb = model.encode(table[0] + '.csv', show_progress_bar=False)
            # 与所有表描述文件做相似度比对
            files_sim = (files_emb @ target_file_emb.T)
            # 若相似度>0.9, 选用最相似的csv（有可能拼写/命名存在误差）
            if max(files_sim) > 0.9:
                file = os.path.join(table_dir, file_list[files_sim.argmax()])
            else:
                file = os.path.join(table_dir, table[0] + '.csv')  # 否则按表名拼接csv路径

            # 读取描述csv（自动判断编码）；如异常则填空DataFrame
            try:
                with open(file, 'rb') as f:
                    result = chardet.detect(f.read())
                table_df = pd.read_csv(file, encoding=result['encoding'])
            except Exception as e:
                print(e)
                table_df = pd.DataFrame()
            # 获取该表完整struct+注释字符串和详细字段字典
            table_info, columns = self.get_complete_table_info(conn, table[0], table_df)
            db_info.append(table_info)
            db_col.update(columns)
        db_info = "\n".join(db_info)
        cursor.close()
        conn.close()
        return db_info, db_col

    # 给定注释：生成数据库总结提示词
    def db_conclusion(self, db_info):
        prompt = f"""/* Here is a examples about describe database */
    #Forigen keys: 
    Airlines.ORIGIN = Airports.Code, Airlines.DEST = Airports.Code, Airlines.OP_CARRIER_AIRLINE_ID = Air Carriers.Code
    #Database Description: The database encompasses information related to flights, including airlines, airports, and flight operations.
    #Tables Descriptions:
    Air Carriers: Codes and descriptive information about airlines
    Airports: IATA codes and descriptions of airports
    Airlines: Detailed information about flights 

    /* Here is a examples about describe database */
    #Forigen keys:
    data.ID = price.ID, production.ID = price.ID, production.ID = data.ID, production.country = country.origin
    #Database Description: The database contains information related to cars, including country, price, specifications, Production
    #Tables Descriptions:
    Country: Names of the countries where the cars originate from.
    Price: Price of the car in USD.
    Data: Information about the car's specifications
    Production: Information about car's production.

    /* Describe the following database */
    {db_info}
    Please conclude the database in the following format:
    #Database Description:
    #Tables Descriptions:
    """
        return prompt

# 给定注释：以字符格式输出表信息的数据库代理
class db_agent_string(db_agent):
    def __init__(self, chat_model) -> None:
        super().__init__(chat_model)

    # 给定注释：获取更详细且带有注释的表信息字符串
    def get_complete_table_info(self, conn, table_name, table_df):
        """
        获取更详细且带有注释的表结构信息，并提取字段描述、示例值、数据特性等信息，便于后续构建数据库schema字符串和字段字典。

        参数:
            conn: SQLite的数据库连接对象。
            table_name: 字符串，表名。
            table_df: DataFrame，包含从额外文件（如表结构csv）读取到的列描述等信息。

        返回:
            schema_str: 字符串，格式化后的该表的结构与字段描述。
            columns: dict，每个字段对应其详细信息（schema结构、描述、类型、唯一性、示例值等）。
        """
        cursor = conn.cursor()
        # 获取该表字段的元信息，包括名字、类型、是否非空、默认值、主键标志等
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        columns_info = cursor.fetchall()

        # 读取整个表的数据
        df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)

        # 统计各个字段是否含有NULL值、是否存在重复值（非唯一性）
        contains_null = {column: df[column].isnull().any() for column in df.columns}
        contains_duplicates = {column: df[column].duplicated().any() for column in df.columns}

        # 解析表外部提供的列信息描述，建立对应关系字典dic：key=列名，value=(列描述，值描述)
        dic = {}
        for _, row in table_df.iterrows():
            try:
                col_description, val_description = "", ""
                col = row.iloc[0].strip()
                # 解析列的功能性描述
                if pd.notna(row.iloc[2]):
                    col_description = re.sub(r'\s+', ' ', str(row.iloc[2]))
                if col_description.strip() == col or col_description.strip() == "":
                    col_description = ''
                # 解析列的取值样式描述
                if pd.notna(row.iloc[4]):
                    val_description = re.sub(r'\s+', ' ', str(row.iloc[4]))
                if val_description.strip() == "" or val_description.strip() == col or val_description == col_description:
                    val_description = ""
                # 限制描述最大长度200
                col_description = col_description[:200]
                val_description = val_description[:200]
                dic[col] = col_description, val_description
            except Exception as e:
                print(e)
                dic[col] = "", ""

        # 采集部分典型行数据作为字段示例值（适合后续给出Values信息）
        # 先尝试取表的第一行样本
        try:
            row = list(cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 1").fetchall()[0])
        except Exception as e:
            # 如果表为空，则用None占位
            row = [None for _ in df.columns]

        # 遍历所有字段，采集每列的典型取值（sample最多3个不同的非缺失值）
        for i, col in enumerate(df.columns):
            try:
                df_tmp = df[col].dropna().drop_duplicates()
                if len(df_tmp) >= 3:
                    vals = df_tmp.sample(3).values
                else:
                    vals = df_tmp.values
                val_p = []
                for val in vals:
                    try:
                        val_p.append(int(val))
                    except:
                        val_p.append(val)
                if len(vals) == 0:
                    raise ValueError
                row[i] = val_p
            except:
                # 若采样失败，则保持默认
                pass

        # 构建最终的schema字符串
        schema_str = f"## Table {table_name}:\n"
        columns = {}
        # 遍历列元信息和采样值，逐列构建标准化schema描述
        for column, val in zip(columns_info, row):
            schema_str_single = ""
            column_name, column_type, not_null, default_value, pk = column[1:6]
            tmp_col = column_name.strip()
            column_name = quote_field(column_name)
            col_des, val_des = dic.get(tmp_col, ["", ""])
            # 拼接字段本身、取值特征、类型、是否含空、唯一性、典型值等
            if col_des != "":
                schema_str_single += f" The column is {col_des}. "
            if val_des != "":
                schema_str_single += f" The values' format are {val_des}. "
            schema_str_single += f"The type is {column_type}, "
            if contains_null[tmp_col]:
                schema_str_single += f"Which inlude Null"
            else:
                schema_str_single += f"Which does not inlude Null"
            if contains_duplicates[tmp_col]:
                schema_str_single += " and is Non-Unique. "
            else:
                schema_str_single += " and is Unique. "
            # 便于结构化表述存储唯一性/是否含Null
            include_null = f"{'Include Null' if contains_null[tmp_col] else 'Non-Null'}"
            unique = f"{'Non-Unique' if contains_duplicates[tmp_col] else 'Unique'}"
            # 采样值展示逻辑（过长则提示为Long text，较小样本直接罗列）
            if len(str(val)) > 360:
                val = "<Long text>"
                schema_str_single += f"Values format: <Long text>"
            elif type(val) is not list or len(val) < 3:
                schema_str_single += f"Value of this column must in: {val}"
            else:
                schema_str_single += f"Values format like: {val}"
            # 加入本字段描述到表结构字符串
            schema_str += f"{column_name}: {schema_str_single}\n"
            # 记录详细字段信息进dict

            # 这些值依次代表：
            # 列描述和值描述都是csv文件里面给的，不一定都有
            # 1. `schema_str_single`：每个字段的详细结构描述字符串（含类型、唯一性、空值等说明）
            # 2. `col_des`：字段（列）的文本描述（如“订单编号”）
            # 3. `val_des`：该字段典型值的格式或示例描述
            # 4. `column_type`：字段的数据类型（如INTEGER, TEXT等）
            # 5. `include_null`：是否允许NULL，取值为"Include Null"/"Non-Null"
            # 6. `unique`：唯一性，"Unique"或"Non-Unique"
            # 7. `str(val)`：示例值（或值的样例列表字符串化结果）
            columns[f"{table_name}.{column_name}"] = (
                schema_str_single,
                col_des,
                val_des,
                column_type,
                include_null,
                unique,
                str(val)
            )
        return schema_str, columns