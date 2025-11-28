from llm import all_prompt

# extract_prompt ：用于从自然语言查询中提取结构化信息
# new_prompt ：用于生成 SQL 查询
# correct_prompt ：用于改进和修复生成的 SQL 中的错误
# db_check_prompts ：添加用于数据库验证和投票的专门提示
# sft_prompts ：专为精细调整模型而设计
# prompts_wo_hint_* 类：不同的无提示支持策略 专门用于没有明确提示的场景
# noun_prompt：用于从问题中提取名词
# select_prompt:从多个SQL候选集中进行选择
# vote_prompt:从多个SQL候选集中投票（自一致性）
# parse_fewshot:


# 该prompt用于根据question和sql生成带有具体格式的fewshots解析格式如reason、columns、select、values、SQL-like
class prompts_fewshot_parse:## fewshot parse prompt
    parse_fewshot= all_prompt.prompts_fewshot_parse2

# 整个系统的核心提示
# 基本提示词基类。包含Extract、Generation、Correction
class prompts1:
    extract_prompt= all_prompt.extract_prompt
    new_prompt=all_prompt.new_prompt0
    correct_prompt=all_prompt.correct_prompt

class prompts_wo_hint_only_sqllike_reparse_ext_atom(prompts1):# deepseek 68  qwenmax 61  全量：
    new_prompt=all_prompt.new_prompt2
    extract_prompt=all_prompt.reparse_extract_prompt
    noun_prompt=all_prompt.noun_prompt
    correct_prompt=all_prompt.correct_prompt_wo_hint


class prompts_wo_hint_only_sqllike_reparse_ext_atom_step(prompts1):# deepseek 68  qwenmax 61  全量：
    new_prompt=all_prompt.new_prompt3
    tmp_prompt=all_prompt.new_prompt1
    extract_prompt=all_prompt.reparse_extract_prompt
    noun_prompt=all_prompt.noun_prompt
    correct_prompt=all_prompt.correct_prompt_wo_hint
    soft_prompt=all_prompt.soft_prompt

class db_check_prompts(prompts_wo_hint_only_sqllike_reparse_ext_atom_step):
    extract_prompt=all_prompt.new_extract_prompt
    extract_prompt_wofewshot=all_prompt.new_extract_prompt_wofewshot
    new_prompt=all_prompt.new_prompt_O
    new_prompt_wocot=all_prompt.new_prompt_O_wocot
    new_prompt_uns_cot=all_prompt.new_prompt_unstruct_cot
    tmp_prompt=all_prompt.new_prompt3
    tmp_prompt_wocot=all_prompt.new_prompt3_wocot
    select_prompt=all_prompt.select_prompt
    vote_prompt=all_prompt.vote_prompt

class sft_prompts(prompts_wo_hint_only_sqllike_reparse_ext_atom_step):
    new_prompt=all_prompt.new_prompt1
    
class prompts_wo_hint_only_sqllike_reparse_ext(prompts1):#
    new_prompt=all_prompt.new_prompt1
    extract_prompt=all_prompt.reparse_extract_prompt
    noun_prompt=all_prompt.noun_prompt

class prompts_wo_hint_no_sqllike(prompts1):#5
    new_prompt=all_prompt.new_prompt_wo_hint_standQ_newsqllike

class prompts_wo_hint_no_sqllike5(prompts1):#57
    new_prompt=all_prompt.new_prompt_wo_hint_standQ
    
class prompts_wo_hint_only_sqllike(prompts1):#58 dev 57.89  #当前sota
    new_prompt=all_prompt.new_prompt1

class prompts_wo_hint_no_sqllike4(prompts1):#55
    extract_prompt= all_prompt.extract_prompt_wo_hint_no_Sqllike
    new_prompt=all_prompt.new_prompt_wo_hint_new_sqllike

class prompts_wo_hint_no_sqllike3(prompts_wo_hint_no_sqllike):
    extract_prompt= all_prompt.extract_prompt_wo_hint_no_Sqllike
    new_prompt=all_prompt.new_prompt_wo_hint_new_sqllike_wodb
    correct_prompt=all_prompt.correct_prompt_nodb

class prompts_wo_hint_no_sqllike_2(prompts_wo_hint_no_sqllike):
    new_prompt=all_prompt.new_prompt_wo_hint_no_sqllike
    