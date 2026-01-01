import logging
from typing import Any, Dict, List
from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from pipeline.utils import make_newprompt
from llm.model import model_chose
from llm.db_conclusion import *
import json
from llm.prompts import *
from runner.check_and_correct import get_sql
from concurrent.futures import ThreadPoolExecutor, as_completed

class DivideConquerGenerator:
    """
    LLM-powered Divide and Conquer Chain-of-Thought generator.
    Adapted to use the project's chat_model.
    """
    
    def __init__(self, chat_model, L_values: List[Any], fewshot: str = ""):
        self.chat_model = chat_model
        self.L_values = L_values
        self.fewshot = fewshot
    
    def generate_candidate(self, question: str, schema: str) -> Dict[str, Any]:
        """Generate SQL using LLM-powered divide and conquer approach."""
        
        # Step 1: LLM-powered question analysis and decomposition
        decomposition_result = self._llm_decompose_question(question, schema)
        
        # Step 2: Generate sub-solutions using LLM
        sub_solutions = self._llm_solve_subproblems(decomposition_result, schema)
        
        # Step 3: Combine solutions using LLM
        final_sql = self._llm_combine_solutions(question, sub_solutions, schema)
        
        return {
            'sql': final_sql,
            'reasoning': {
                'approach': 'llm_divide_and_conquer',
                'decomposition': decomposition_result,
                'sub_solutions': sub_solutions,
                'llm_reasoning': 'Used Chain-of-Thought decomposition'
            }
        }
    
    def _llm_decompose_question(self, question: str, schema: str) -> Dict[str, Any]:
        """Use LLM to decompose the question into sub-problems."""
        
        # Get some context from value retrieval
        context = self._format_value_context()
        
        prompt = f"""You are an expert at breaking down complex SQL questions into simpler sub-problems.

Database Schema:
{schema}

Context (relevant values found in database):
{context}

Question: {question}

Please analyze this question and break it down using divide-and-conquer approach. Think step by step:

1. ANALYSIS: What is the main intent? What entities are involved? What constraints/filters are needed?
2. DECOMPOSITION: Break this into 2-4 logical sub-problems that can be solved independently
3. DEPENDENCIES: What order should these sub-problems be solved in?

Format your response as:
ANALYSIS:
- Main Intent: [count/retrieve/aggregate/etc]
- Entities: [tables involved]  
- Constraints: [filters needed]
- Aggregations: [any aggregation functions]

DECOMPOSITION:
1. [Sub-problem 1 description]
2. [Sub-problem 2 description]
3. [Sub-problem 3 description if needed]

DEPENDENCIES:
[Explain the logical order and dependencies between sub-problems]"""

        try:
            # Using chat_model.get_ans with single=True to get string content
            response_text = self.chat_model.get_ans(prompt, single=True)
            return self._parse_decomposition_response(response_text, question)
        except Exception as e:
            logging.error(f"Error in decomposition: {e}")
            return self._fallback_analyze_question(question)
    
    def _llm_solve_subproblems(self, decomposition: Dict[str, Any], schema: str) -> List[Dict[str, Any]]:
        """Use LLM to solve each sub-problem."""
        
        sub_solutions = []
        
        for i, sub_problem in enumerate(decomposition.get('sub_problems', [])):
            prompt = f"""You are solving sub-problem {i+1} of a divide-and-conquer SQL generation task.

Database Schema:
{schema}

Overall Question Context: {decomposition.get('original_question', '')}
Analysis: {decomposition.get('analysis', {})}

Sub-problem to solve: {sub_problem}

Previous sub-solutions for context:
{self._format_previous_solutions(sub_solutions)}

Generate the SQL component or logic needed for this specific sub-problem. 
Focus ONLY on this sub-problem, not the complete query.

Respond with:
APPROACH: [How you're solving this sub-problem]
SQL_COMPONENT: [The SQL piece for this sub-problem]
EXPLANATION: [Brief explanation of this component]"""

            try:
                response_text = self.chat_model.get_ans(prompt, single=True)
                solution = self._parse_subproblem_response(response_text, sub_problem)
                sub_solutions.append(solution)
            except Exception as e:
                logging.error(f"Error in solving subproblem: {e}")
                sub_solutions.append({
                    'sub_problem': sub_problem,
                    'approach': 'fallback',
                    'sql_component': 'SELECT * FROM table',
                    'explanation': 'Fallback solution'
                })
        
        return sub_solutions
    
    def _llm_combine_solutions(self, question: str, sub_solutions: List[Dict], schema: str) -> str:
        """Use LLM to combine sub-solutions into final SQL."""
        
        solutions_text = "\n".join([
            f"Sub-problem: {sol['sub_problem']}\nSQL Component: {sol['sql_component']}\nApproach: {sol['approach']}"
            for sol in sub_solutions
        ])
        
        prompt = f"""You are combining sub-solutions into a complete SQL query.

Database Schema:
{schema}

Original Question: {question}

Sub-solutions to combine:
{solutions_text}

Now combine these sub-solutions into a single, complete, executable SQL query.
Make sure to:
1. Use proper JOIN syntax if multiple tables are involved
2. Apply all necessary WHERE conditions  
3. Use correct aggregation functions
4. Ensure proper ORDER BY and LIMIT clauses where needed
5. Make the query syntactically correct

Return ONLY the final SQL query, no explanation."""

        try:
            response_text = self.chat_model.get_ans(prompt, single=True)
            sql = response_text.strip()
            
            # Clean up the response
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.startswith('```'):
                sql = sql[3:]
            if sql.endswith('```'):
                sql = sql[:-3]
            return sql.strip()
            
        except Exception as e:
            logging.error(f"Error in combine solutions: {e}")
            return 'SELECT COUNT(*) FROM customers' # simple fallback

    def _format_value_context(self) -> str:
        """Format relevant values for context."""
        if not self.L_values:
            return "No specific values found in database for this question."
        
        lines = []
        # L_values is list of [column, value]
        for item in self.L_values[:10]:  # Limit to top 10 to avoid too much context
            if len(item) >= 2:
                lines.append(f"- '{item[0]}' has value '{item[1]}'")
        
        return "\n".join(lines) if lines else "No specific value mappings found."
    
    def _parse_decomposition_response(self, response: str, question: str) -> Dict[str, Any]:
        """Parse LLM decomposition response."""
        
        result = {
            'original_question': question,
            'analysis': {},
            'sub_problems': [],
            'dependencies': ''
        }
        
        if not response:
            return self._fallback_analyze_question(question)

        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('ANALYSIS:'):
                current_section = 'analysis'
            elif line.startswith('DECOMPOSITION:'):
                current_section = 'decomposition'
            elif line.startswith('DEPENDENCIES:'):
                current_section = 'dependencies'
            elif line and current_section:
                if current_section == 'decomposition' and (line[0].isdigit() and line[1] == '.'):
                    result['sub_problems'].append(line[2:].strip())
                elif current_section == 'dependencies':
                    result['dependencies'] += line + ' '
                elif current_section == 'analysis' and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key, value = parts
                        result['analysis'][key.strip('- ')] = value.strip()
        
        if not result['sub_problems']:
             return self._fallback_analyze_question(question)
        
        return result
    
    def _parse_subproblem_response(self, response: str, sub_problem: str) -> Dict[str, Any]:
        """Parse LLM sub-problem response."""
        
        result = {
            'sub_problem': sub_problem,
            'approach': '',
            'sql_component': '',
            'explanation': ''
        }
        
        if not response:
            return result

        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('APPROACH:'):
                current_section = 'approach'
                result['approach'] = line[9:].strip()
            elif line.startswith('SQL_COMPONENT:'):
                current_section = 'sql_component'
                result['sql_component'] = line[14:].strip()
            elif line.startswith('EXPLANATION:'):
                current_section = 'explanation'
                result['explanation'] = line[12:].strip()
            elif line and current_section:
                result[current_section] += ' ' + line
        
        sql = result['sql_component']
        if sql.startswith('```sql'):
            sql = sql[6:]
        if sql.endswith('```'):
            sql = sql[:-3]
        result['sql_component'] = sql.strip()
        
        return result
    
    def _format_previous_solutions(self, solutions: List[Dict]) -> str:
        """Format previous solutions for context."""
        if not solutions:
            return "No previous solutions yet."
        
        formatted = []
        for i, sol in enumerate(solutions):
            formatted.append(f"{i+1}. {sol['sub_problem']}: {sol.get('sql_component', 'N/A')}")
        
        return "\n".join(formatted)

    def _fallback_analyze_question(self, question: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        return {
            'original_question': question,
            'analysis': {'main_intent': 'retrieve'},
            'sub_problems': [
                "Identify target tables",
                "Apply filters",
                "Format output"
            ],
            'dependencies': 'Sequential'
        }

def _run_fewshot(chat_model, new_prompt, config):
    """Helper to run fewshot generation in a thread."""
    return_question = config['return_question'] == 'true'
    return get_sql(
        chat_model,
        new_prompt,
        config['temperature'],
        return_question=return_question,
        n=1,
        single=True
    )

@node_decorator(check_schema_status=False)
def candidate_generate(task: Any, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate SQL candidate set node:
    1. Read few-shot examples.
    2. Integrate upstream node information.
    3. Construct prompts.
    4. Call LLM to generate SQL candidates (Fewshot + Divide & Conquer).
    5. Return result structure.
    """
    # Get config and current node name
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()
    fewshot_path = paths.db_fewshot_path

    # ===== 1. Load few-shot examples =====
    with open(fewshot_path, 'r') as f:
        df_fewshot = json.load(f)

    # ===== 2. Initialize LLM and prepare context =====
    chat_model = model_chose(node_name, config["engine"])
    column = get_last_node_result(execution_history, "column_retrieve_and_other_info")["column"]
    foreign_keys = get_last_node_result(execution_history, "column_retrieve_and_other_info")["foreign_keys"]
    L_values = get_last_node_result(execution_history, "column_retrieve_and_other_info")["L_values"]
    q_order = get_last_node_result(execution_history, "column_retrieve_and_other_info")["q_order"]
    values = [f"{x[0]}: '{x[1]}'" for x in L_values]
    db = task.db_id

    # ===== 3. Organize key info for prompt =====
    key_col_des = "#Values in Database:\n" + '\n'.join(values)
    new_db_info = (
        f"Database Management System: SQLite\n"
        f"#Database name: {db} \n"
        f"{column}\n\n"
        f"#Forigen keys:\n{foreign_keys}\n"
    )

    question = task.question
    fewshot_raw = df_fewshot["questions"][task.question_id]['prompt']

    # ===== 4. Construct prompt for Few-shot generator =====
    new_prompt = make_newprompt(
        db_check_prompts().new_prompt,
        fewshot_raw,
        key_col_des,
        new_db_info,
        question,
        task.evidence,
        q_order
    )

    # ===== 5. Concurrent Generation =====
    n_gen = int(config.get('n', 1))
    
    sqls = []
    rewrite_q_final = question # Default to original question

    # Initialize Divide Conquer Generator
    dc_generator = DivideConquerGenerator(chat_model, L_values, fewshot_raw)

    with ThreadPoolExecutor(max_workers=n_gen * 2) as executor:
        futures = []
        
        # Submit Few-shot tasks
        for _ in range(n_gen):
            futures.append(executor.submit(_run_fewshot, chat_model, new_prompt, config))
            
        # Submit Divide & Conquer tasks
        for _ in range(n_gen):
            futures.append(executor.submit(dc_generator.generate_candidate, question, new_db_info))
            
        for future in as_completed(futures):
            try:
                res = future.result()
                if isinstance(res, tuple): # Result from _run_fewshot
                    sql, rq = res
                    if sql: 
                        sqls.append(sql)
                    if rq: 
                        rewrite_q_final = rq
                elif isinstance(res, dict): # Result from DivideConquerGenerator
                    sql = res.get('sql')
                    if sql:
                        sqls.append(sql)
            except Exception as e:
                logging.error(f"Error in candidate generation future: {e}")

    # ===== 6. Fallback and Output =====
    if not sqls:
        raise ValueError("LLM API call failed in candidate_generate. No SQLs generated.")

    response = {
        "rewrite_question": rewrite_q_final,
        "SQL": sqls
    }

    return response

def rewrite_question(question):
    """
    For questions involving division/float display etc, add precision hint.
    """
    if question.find(" / ") != -1:
        question += ". For division operations, use CAST xxx AS REAL to ensure precise decimal results"
    return question
