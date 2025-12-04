# Define variables
data_mode='dev' # Options: 'dev', 'train' 
db_root_path=Bird #root directory # UPDATE THIS WITH THE PATH TO THE TARGET DATASET
start=0 #闭区间
end=1  #开区间
pipeline_nodes='generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation'
# pipeline指当前工作流的节点组合
# checkpoint_nodes='generate_db_schema,extract_col_value,extract_query_noun'
# checkpoint_dir="./results/dev/generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation/Bird/2024-09-12-01-48-10"

# Nodes:
    # generate_db_schema
    # extract_col_value
    # extract_query_noun
    # column_retrieve_and_other_info
    # candidate_generate
    # align_correct
    # vote
    # evaluation

# engine1='gpt-4o-0513'
engine1= 'qwen'
engine2='gpt-3.5-turbo-0125'
engine3='gpt-4-turbo'
engine4='claude-3-opus-20240229'
engine5='gemini-1.5-pro-latest'
engine6='finetuned_nl2sql'
engine7='meta-llama/Meta-Llama-3-70B-Instruct'
engine8='finetuned_colsel'
engine9='finetuned_col_filter'
engine10='gpt-3.5-turbo-instruct'
 
pipeline_setup='{
    "generate_db_schema": {
        "engine": "'${engine1}'",
        "bert_model": "your_bert_model_path",  
        "device":"cpu"
    },
    "extract_col_value": {
        "engine": "'${engine1}'",
        "temperature":0.0
    },
    "extract_query_noun": {
        "engine": "'${engine1}'",
        "temperature":0.0
    },
    "column_retrieve_and_other_info": {
        "engine": "'${engine1}'",
        "bert_model": "your_bert_model_path",  
        "device":"cpu",
        "temperature":0.3,
        "top_k":10
    },
    "candidate_generate":{
        "engine": "'${engine1}'",
        "temperature": 0.7,  
        "n":7,
        "return_question":"True",
        "single":"False"
    },
    "align_correct":{
        "engine": "'${engine1}'",
        "n":7,
        "bert_model": "your_bert_model_path:e.g. /opensearch-sql/bge",  
        "device":"cpu",
        "align_methods":"style_align+function_align+agent_align"
    }
}'  

python3 -u ./src/main.py --data_mode ${data_mode} --db_root_path ${db_root_path}\
        --pipeline_nodes ${pipeline_nodes} --pipeline_setup "$pipeline_setup"\
        --start ${start} --end ${end} \
        # --use_checkpoint --checkpoint_nodes ${checkpoint_nodes} --checkpoint_dir ${checkpoint_dir}
  
