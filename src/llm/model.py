import requests, time
import dashscope
import torch
import json
import re
from runner.logger import Logger
from llm.prompts import prompts_fewshot_parse

# 选择不同大模型调用方式，根据传入参数选择gpt/qwen/deepseek/sft等
# 最终返回一个LLM实例对象
def model_chose(step, model="gpt-4 32K"):
    # GPT与Claude/Gemini等API
    if model.startswith("gpt") or model.startswith("claude35_sonnet") or model.startswith("gemini"):
        return gpt_req(step, model)
    # DeepSeek API
    if model == "deepseek":
        return deep_seek(model)
    # QwenMax API
    if model.startswith("qwen"):
        return qwenmax(model)
    # SFT本地微调模型
    if model.startswith("sft"):
        return sft_req()

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

# 通用请求方法，适配多模型API接口，仅用于gpt_req
def request(url, model, messages, temperature, top_p, n, key, **k):
    res = requests.post(
        url=url,
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an SQL expert, skilled in handling various SQL-related issues."
                },
                {
                    "role": "user",
                    "content": messages
                }
            ],
            "max_tokens": 800,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            **k
        },
        headers={
            "Authorization": key
        }
    ).json()
    return res

# OpenAI 的 GPT 模型、Anthropic 的 Claude 和 Google 的 Gemini 的请求：
class gpt_req(req):
    def __init__(self, step, model="gpt-4o-0513") -> None:
        super().__init__(step, model)

    # 核心请求方法，支持重试，并返回结果
    def get_ans(self, messages, temperature=0.0, top_p=None, n=1, single=True, **k):
        count = 0
        while count < 50:
            try:
                res = request(
                    url="",
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    n=n, key="",
                    **k
                )
                if n == 1 and single:
                    response_clean = res["choices"][0]["message"]["content"]
                else:
                    response_clean = res["choices"]
                # 仅prepare_train_queries阶段不记录日志
                if self.step != "prepare_train_queries":
                    self.log_record(messages, response_clean)
                break
            except Exception as e:
                count += 1
                time.sleep(2)
                print(e, count, self.Cost, res)
        # 记录token消耗
        self.Cost += res["usage"]['prompt_tokens'] / 1000 * 0.042 + res["usage"]["completion_tokens"] / 1000 * 0.126
        return response_clean

# DeepSeek大模型API请求封装
class deep_seek(req):
    def __init__(self, model) -> None:
        super().__init__(model)

    def get_ans(self, messages, temperature=0.0, debug=False):
        count = 0
        while count < 8:
            try:
                url = "https://api.deepseek.com/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": ""
                }
                # 定义请求体
                jsons = {
                    "model": "deepseek-coder",
                    "temperture": temperature,  # 注意：写错温度字段名
                    "top_p": 0.9,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": messages}
                    ]
                }
                # 发送POST请求
                response = requests.post(url, headers=headers, json=jsons)
                if debug:
                    print(response.json)
                ans = response.json()['choices'][0]['message']['content']
                break
            except Exception as e:
                count += 1
                time.sleep(2)
                print(e, count, self.Cost, response.json())
        return ans

# Qwen Max API
class qwenmax(req):
    def __init__(self, model) -> None:
        super().__init__(model)
        # 在此处填写你的Qwen API Key
        self.api_key = ""

    def get_ans(self, messages, temperature=0.0, debug=False):
        count = 0
        ans = None
        url = "https://api.aliyun.com/qwen/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        while count < 8:
            try:
                payload = {
                    "model": self.model,  # 例如："qwen-32b-chat"
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI assistant able to answer SQL or code-related questions."
                        },
                        {
                            "role": "user",
                            "content": messages
                        }
                    ],
                    "temperature": temperature,
                    "top_p": 0.9,
                    "max_tokens": 800,
                }
                response = requests.post(url, headers=headers, json=payload)
                if debug:
                    print(response.text)
                response_json = response.json()
                ans = response_json["choices"][0]["message"]["content"]
                # 统计token信息，如果接口返回了usage
                if "usage" in response_json:
                    usage = response_json["usage"]
                    self.Cost += usage.get("prompt_tokens", 0) / 1000 * 0.04 + usage.get("completion_tokens", 0) / 1000 * 0.12
                break
            except Exception as e:
                count += 1
                time.sleep(5)
                print(e)
        return ans

# SFT本地自定义微调大模型推理接口
class sft_req(req):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.device = "cuda:0"
        # 加载本地Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "",
            trust_remote_code=True,
            padding_side="right",
            use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token = "<|EOT|>"
        # 加载本地推理模型
        self.model = AutoModelForCausalLM.from_pretrained(
            "",
            torch_dtype=torch.bfloat16,
            device_map=self.device).eval()

    # 本地模型推理
    def get_ans(self, text, temperature=0.0):
        messages = [{
            "role": "system",
            "content": (
                "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, "
                "and you only answer questions related to computer science. For politically sensitive questions, "
                "security and privacy issues, and other non-computer science questions, you will refuse to answer."
            )
        }, {
            "role": "user",
            "content": text
        }]
        # 处理prompt生成inputs
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False)
        model_inputs = self.tokenizer([inputs],
                                      return_tensors="pt",
                                      max_length=8000).to("cuda")
        # tokenizer.eos_token_id is the id of <|EOT|> token
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=800,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.decode(generated_ids[0][:-1],
                                         skip_special_tokens=True).strip()
        return response
