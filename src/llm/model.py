import requests, time
import dashscope
import torch
import json
import re
import os
from runner.logger import Logger
from llm.prompts import prompts_fewshot_parse

# todo add api_key
# 选择不同大模型调用方式，根据传入参数选择gpt/qwen/deepseek/sft等
# 最终返回一个LLM实例对象
def model_chose(step, model="gpt-4 32K"):
    # GPT与Claude/Gemini等API
    if model.startswith("gpt") or model.startswith("claude35_sonnet") or model.startswith("gemini"):
        return gpt_req(step, model)
    # DeepSeek API
    if model == "deepseek":
        return deep_seek(step,model)
    # QwenMax API
    if model.startswith("qwen"):
        return qwen(step,model)
    # # SFT本地微调模型
    # if model.startswith("sft"):
    #     return sft_req()
    else:
        raise ValueError(f"Unsupported model: {model}")

# 各大模型统一的基类：req
class req:
    # 初始化，记录Cost消耗、模型名、调用阶段
    def __init__(self, step, model) -> None:
        self.Cost = 0
        self.model = model
        self.step = step

    # 日志记录prompt和输出
    def log_record(self, prompt_text, output):
        logger = Logger()
        logger.log_conversation(prompt_text, "Human", self.step)
        logger.log_conversation(output, "AI", self.step)

    # fewshot_parse，用于少样本问题解析调用
    def fewshot_parse(self, question, evidence, sql):
        s = prompts_fewshot_parse().parse_fewshot.format(question=question, sql=sql)
        ext = self.get_ans(s)
        ext = ext.replace('```', '').strip()
        ext = ext.split("#SQL:")[0]  # 保底SQL，不符合格式也能取出SQL部分
        ans = self.convert_table(ext, sql)
        return ans

    # 处理模型输出，修正表别名
    def convert_table(self, s, sql):
        l = re.findall(' ([^ ]*) +AS +([^ ]*)', sql)
        x, v = s.split("#values:")
        t, s = x.split("#SELECT:")
        for li in l:
            s = s.replace(f"{li[1]}.", f"{li[0]}.")
        return t + "#SELECT:" + s + "#values:" + v

# 通用请求方法，适配多模型API接口
def request(url, model, messages, temperature, top_p, n, key, **k):
    """
    重构：统一使用 OpenAI 兼容格式 (目前 DeepSeek, Qwen, GPT 都支持此格式)
    """
    # 构造标准 OpenAI 格式 Messages
    if isinstance(messages, str):
        msg_list = [
            {"role": "system", "content": "You are an SQL expert."},
            {"role": "user", "content": messages}
        ]
    else:
        msg_list = messages

    body = {
        "model": model,
        "messages": msg_list,
        "max_tokens": 800,
        "temperature": temperature,
        "n": n,
        **k # 允许传入 stream 等其他参数
    }
    
    # 只有当 top_p 不为 None 时才加入，避免某些 API 报错
    if top_p is not None:
        body["top_p"] = top_p

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url=url, json=body, headers=headers, timeout=60)
        response.raise_for_status() # 检查 HTTP 状态码
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"JSON decode failed or other error: {e}")
        return None

# OpenAI 的 GPT 模型、Anthropic 的 Claude 和 Google 的 Gemini 的请求：
class gpt_req(req):
    def __init__(self, step, model="gpt-4o") -> None:
        super().__init__(step, model)
        self.api_key = ""
        # 填入正确的 URL
        self.url = "https://api.openai.com/v1/chat/completions"

    def get_ans(self, messages, temperature=0.0, top_p=None, n=1, single=True, **k):
        return self._unified_get_ans(messages, temperature, top_p, n, single, 
                                     price_prompt=0.042, price_completion=0.126, **k)

    # 提取公共重试逻辑到父类或辅助方法是个好习惯，这里暂时放在类内
    def _unified_get_ans(self, messages, temperature, top_p, n, single, price_prompt, price_completion, **k):
        count = 0
        while count < 8:
            res = request(
                url=self.url,
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                n=n,
                key=self.api_key,
                **k
            )
            
            if res and "choices" in res:
                if "usage" in res:
                    self.Cost += res["usage"].get('prompt_tokens', 0) / 1000 * price_prompt + \
                                 res["usage"].get("completion_tokens", 0) / 1000 * price_completion
                
                output = res["choices"]
                if n == 1 and single:
                    content = output[0]["message"]["content"]
                    if self.step != "prepare_train_queries":
                        self.log_record(messages, content)
                    return content
                return output
            
            # 如果失败
            count += 1
            print(f"Retry {count}: {res}")
            time.sleep(2)
        return None

# DeepSeek大模型API请求封装
class deep_seek(gpt_req): # 继承 gpt_req 复用逻辑
    def __init__(self, step, model="deepseek-chat") -> None:
        # 注意：这里调用的是 req 的 init，因为 gpt_req 的 init 硬编码了 url
        req.__init__(self, step, model) 
        self.api_key = ""
        self.url = "[https://api.deepseek.com/chat/completions](https://api.deepseek.com/chat/completions)"

    def get_ans(self, messages, temperature=0.0, top_p=None, n=1, single=True, **k):
        # DeepSeek 价格便宜很多
        return self._unified_get_ans(messages, temperature, top_p, n, single, 
                                     price_prompt=0.001, price_completion=0.002, **k)

# Qwen Max API请求封装，适配request函数
class qwen(gpt_req): # 继承 gpt_req 复用逻辑
    def __init__(self, step, model="qwen") -> None:
        req.__init__(self, step, model)
        self.api_key = ""
        self.url = "[https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions](https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions)"

    def get_ans(self, messages, temperature=0.0, top_p=None, n=1, single=True, **k):
        return self._unified_get_ans(messages, temperature, top_p, n, single, 
                                     price_prompt=0.04, price_completion=0.12, **k)

# # SFT本地自定义微调大模型推理接口
# class sft_req(req):
#     def __init__(self, model) -> None:
#         super().__init__(model)
#         self.device = "cuda:0"
#         # 加载本地Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "",
#             trust_remote_code=True,
#             padding_side="right",
#             use_fast=True)
#         self.tokenizer.pad_token = self.tokenizer.eos_token = "<|EOT|>"
#         # 加载本地推理模型
#         self.model = AutoModelForCausalLM.from_pretrained(
#             "",
#             torch_dtype=torch.bfloat16,
#             device_map=self.device).eval()

#     # 本地模型推理
#     def get_ans(self, text, temperature=0.0):
#         messages = [{
#             "role": "system",
#             "content": (
#                 "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, "
#                 "and you only answer questions related to computer science. For politically sensitive questions, "
#                 "security and privacy issues, and other non-computer science questions, you will refuse to answer."
#             )
#         }, {
#             "role": "user",
#             "content": text
#         }]
#         # 处理prompt生成inputs
#         inputs = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=False)
#         model_inputs = self.tokenizer([inputs],
#                                       return_tensors="pt",
#                                       max_length=8000).to("cuda")
#         # tokenizer.eos_token_id is the id of <|EOT|> token
#         generated_ids = self.model.generate(
#             model_inputs.input_ids,
#             attention_mask=model_inputs["attention_mask"],
#             max_new_tokens=800,
#             do_sample=False,
#             eos_token_id=self.tokenizer.eos_token_id,
#             pad_token_id=self.tokenizer.pad_token_id)
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(
#                 model_inputs.input_ids, generated_ids)
#         ]
#         response = self.tokenizer.decode(generated_ids[0][:-1],
#                                          skip_special_tokens=True).strip()
#         return response
