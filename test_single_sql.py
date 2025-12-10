import argparse
import json
import os
import sys
import logging
import pprint
from datetime import datetime
from typing import Any, Tuple, Dict
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add src to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from runner.run_manager import RunManager
from runner.database_manager import DatabaseManager
from runner.logger import Logger
from pipeline.pipeline_manager import PipelineManager
from pipeline.workflow_builder import build_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class TestRunManager(RunManager):
    def worker(self, task) -> Tuple[Any, str, int]:
        """
        Modified worker to print pipeline node outputs and log everything to file.
        """
        # Create logger first
        logger = Logger(db_id=task.db_id, question_id=task.question_id, result_directory=self.result_directory)
        logger._set_log_level(self.args.log_level)
        
        # Create a log file path for comprehensive logging
        # Convert result_directory to Path if it's a string
        from pathlib import Path
        result_dir = Path(self.result_directory) if isinstance(self.result_directory, str) else self.result_directory
        log_file_path = result_dir / "logs" / f"{task.question_id}_{task.db_id}_full.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Function to write to both console and file
        def log_to_both(message, to_file=True):
            print(message)  # Always print to console
            if to_file:
                with log_file_path.open("a", encoding='utf-8') as f:
                    f.write(message + "\n")
        
        # Start logging
        log_to_both(f"\n{'='*80}")
        log_to_both(f"Processing Task - DB: {task.db_id}, Question ID: {task.question_id}")
        log_to_both(f"Question: {task.question}")
        log_to_both(f"Evidence: {task.evidence}")
        log_to_both(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_both(f"{'='*80}\n")

        database_manager = DatabaseManager(db_mode=self.args.data_mode, db_root_path=self.args.db_root_path, db_id=task.db_id)
        
        pipeline_manager = PipelineManager(json.loads(self.args.pipeline_setup))
        execution_history = self.load_checkpoint(task.db_id, task.question_id)

        initial_state = {"keys": {"task": task, "execution_history": execution_history}}
        log_to_both("Building pipeline...")
        self.app = build_pipeline(self.args.pipeline_nodes)
        log_to_both("Pipeline built successfully.\n")

        if hasattr(self.app, 'nodes') and self.app.nodes:
            last_node_key = list(self.app.nodes.keys())[-1]
            log_to_both(f"Pipeline nodes: {', '.join(self.app.nodes.keys())}")
            log_to_both(f"Last node: {last_node_key}\n")
        else:
            last_node_key = None

        final_state = None
        
        # Iterate over the stream and log results
        log_to_both(f"{'='*80}")
        log_to_both("Starting Pipeline Execution")
        log_to_both(f"{'='*80}\n")
        
        for state in self.app.stream(initial_state):
            # state is a dict of updated nodes e.g. {'generate_db_schema': {...}}
            for node_name, output in state.items():
                log_to_both(f"\n{'-'*80}")
                log_to_both(f"Node Output: {node_name}")
                log_to_both(f"{'-'*80}")
                
                # Extract execution history if available
                exec_history = None
                if isinstance(output, dict) and "keys" in output:
                    keys_content = output.get("keys", {})
                    if "execution_history" in keys_content:
                        exec_history = keys_content["execution_history"]
                        
                        # Get the latest execution result
                        if exec_history:
                            latest_result = exec_history[-1]
                            log_to_both(f"Latest execution result:")
                            try:
                                log_to_both(json.dumps(latest_result, indent=2, ensure_ascii=False))
                            except:
                                log_to_both(str(latest_result))
                
                # Log node output using Logger
                if exec_history:
                    logger.log_node_output(node_name, output, exec_history)
                else:
                    logger.log_node_output(node_name, output)
                
                # Pretty print the output to console and file
                try:
                    output_str = pprint.pformat(output, indent=2, width=120)
                    log_to_both(output_str)
                except Exception as e:
                    log_to_both(f"Error formatting output: {e}")
                    log_to_both(str(output))
                
                final_state = state

        log_to_both(f"\n{'='*80}")
        log_to_both("Pipeline Execution Completed")
        log_to_both(f"{'='*80}\n")

        # Log evaluation results if available
        if final_state and last_node_key and last_node_key in final_state:
            final_output = final_state[last_node_key]
            if isinstance(final_output, dict) and "keys" in final_output:
                exec_history = final_output.get("keys", {}).get("execution_history", [])
                # Find evaluation node
                for node_result in exec_history:
                    if node_result.get("node_type") == "evaluation":
                        log_to_both(f"\n{'='*80}")
                        log_to_both("EVALUATION RESULTS")
                        log_to_both(f"{'='*80}")
                        try:
                            eval_str = json.dumps(node_result, indent=2, ensure_ascii=False)
                            log_to_both(eval_str)
                        except:
                            log_to_both(str(node_result))
                        log_to_both(f"{'='*80}\n")
                        break
            
            log_to_both(f"\nFull log saved to: {log_file_path}\n")
            return final_state[last_node_key], task.db_id, task.question_id
        
        log_to_both(f"\nFull log saved to: {log_file_path}\n")
        return final_state, task.db_id, task.question_id

def main():
    parser = argparse.ArgumentParser(description="Run a single SQL test case from the dataset.")
    
    # Default pipeline setup mirroring run_main.sh but with fixed paths
    default_setup = {
        "generate_db_schema": {
            "engine": "qwen-plus",
            "bert_model": "BAAI/bge-m3", 
            "device": "cpu"
        },
        "extract_col_value": {
            "engine": "qwen-plus",
            "temperature": 0.0
        },
        "extract_query_noun": {
            "engine": "qwen-plus",
            "temperature": 0.0
        },
        "column_retrieve_and_other_info": {
            "engine": "qwen-plus",
            "bert_model": "BAAI/bge-m3",
            "device": "cpu",
            "temperature": 0.3,
            "top_k": 10
        },
        "candidate_generate": {
            "engine": "qwen-plus",
            "temperature": 0.7,
            "n": 3,
            "return_question": "True",
            "single": "False"
        },
        "align_correct": {
            "engine": "qwen-plus",
            "n": 3,
            "bert_model": "BAAI/bge-m3",
            "device": "cpu",
            "align_methods": "style_align+function_align+agent_align"
        }
    }

    parser.add_argument('--data_mode', type=str, default='dev', help="Mode of the data to be processed.")
    parser.add_argument('--db_root_path', type=str, default='Bird', help="Path to the data file.")
    parser.add_argument('--pipeline_nodes', type=str, 
                        default='generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation', 
                        help="Pipeline nodes configuration.")
    parser.add_argument('--pipeline_setup', type=str, default=json.dumps(default_setup), help="Pipeline setup in JSON format.")
    parser.add_argument('--use_checkpoint', action='store_true', help="Flag to use checkpointing.")
    parser.add_argument('--checkpoint_nodes', type=str, required=False, help="Checkpoint nodes configuration.")
    parser.add_argument('--checkpoint_dir', type=str, required=False, help="Directory for checkpoints.")
    parser.add_argument('--log_level', type=str, default='info', help="Logging level.")
    
    # Default to running the first case (index 0)
    parser.add_argument('--start', type=int, default=0, help="Start point (index)")
    parser.add_argument('--end', type=int, default=1, help="End point (index)")

    # Custom task arguments
    parser.add_argument('--custom_db_id', type=str, help="Custom database ID (e.g. california_schools)")
    parser.add_argument('--custom_question', type=str, help="Custom question to test")
    parser.add_argument('--custom_evidence', type=str, default="", help="Custom evidence for the question")
    
    args = parser.parse_args()
    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    if args.custom_db_id and args.custom_question:
        print("Using custom task input specified via command line.")
        dataset = [{
            "question_id": 999999,
            "db_id": args.custom_db_id,
            "question": args.custom_question,
            "evidence": args.custom_evidence,
            "SQL": "",
            "difficulty": "custom"
        }]
        # Reset start/end to process this single item
        args.start = 0
        args.end = 1
    else:
        print(f"Loading dataset from {args.db_root_path}...")
        db_json_path = os.path.join(args.db_root_path, 'data_preprocess', f'{args.data_mode}.json')
        
        if not os.path.exists(db_json_path):
            print(f"Error: Dataset file not found at {db_json_path}")
            return

        with open(db_json_path, 'r') as file:
            dataset = json.load(file)
        
        print(f"Loaded {len(dataset)} tasks. Running task(s) from {args.start} to {args.end}...")
    
    run_manager = TestRunManager(args)
    run_manager.initialize_tasks(args.start, args.end, dataset)
    run_manager.run_tasks()

if __name__ == '__main__':
    main()

