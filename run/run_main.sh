#!/bin/bash

# Initialize conda (if not already initialized)
# Try to initialize conda if it's not already in PATH
if ! command -v conda &> /dev/null; then
    # Try common conda installation paths
    if [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi

# Activate conda environment
if command -v conda &> /dev/null; then
    conda activate nl2sql || {
        echo "Error: Failed to activate conda environment 'nl2sql'"
        echo "Trying to use conda environment Python directly..."
        PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/nl2sql/bin/python3"
        if [ ! -f "$PYTHON_BIN" ]; then
            echo "Error: Cannot find Python in conda environment 'nl2sql'"
            exit 1
        fi
        export PYTHON_BIN
    }
else
    # Fallback: use conda environment Python directly
    PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/nl2sql/bin/python3"
    if [ ! -f "$PYTHON_BIN" ]; then
        echo "Error: conda not found and cannot locate conda environment Python"
        exit 1
    fi
    export PYTHON_BIN
fi

# Define variables
data_mode='dev' # Options: 'dev', 'train' 
db_root_path=Bird #root directory # UPDATE THIS WITH THE PATH TO THE TARGET DATASET
start=0 #闭区间
end=4  #开区间
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
engine1='qwen-plus'
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
        "device":"cuda"
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
        "device":"cuda",
        "temperature":0.3,
        "top_k":10
    },
    "candidate_generate":{
        "engine": "'${engine1}'",
        "temperature": 0.7,  
        "n":3,
        "return_question":"True",
        "single":"False"
    },
    "align_correct":{
        "engine": "'${engine1}'",
        "n":3,
        "bert_model": "your_bert_model_path:e.g. /opensearch-sql/bge",  
        "device":"cuda",
        "align_methods":"style_align+function_align+agent_align"
    }
}'  

# Use python3 from the activated conda environment
${PYTHON_BIN:-python3} -u ./src/main.py --data_mode ${data_mode} --db_root_path ${db_root_path}\
        --pipeline_nodes ${pipeline_nodes} --pipeline_setup "$pipeline_setup"\
        --start ${start} --end ${end} \
        # --use_checkpoint --checkpoint_nodes ${checkpoint_nodes} --checkpoint_dir ${checkpoint_dir}
  
