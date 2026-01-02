"""
修复 BGEM3FlagModel 加载问题的 monkey patch
需要在导入 BGEM3FlagModel 之前应用此修复

问题：AutoTokenizer.from_pretrained() 在某些情况下会报错 'dict' object has no attribute 'model_type'
解决：直接使用 XLMRobertaTokenizer 来加载 tokenizer
"""
from transformers import AutoTokenizer, XLMRobertaTokenizer

# 保存原始的 from_pretrained 方法
_original_from_pretrained = AutoTokenizer.from_pretrained

def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    """修复后的 from_pretrained 方法"""
    # 如果是本地路径且是 bge-m3 模型，直接使用 XLMRobertaTokenizer
    if isinstance(pretrained_model_name_or_path, str) and 'bge-m3' in pretrained_model_name_or_path:
        try:
            # 先尝试直接使用 XLMRobertaTokenizer
            return XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except Exception:
            # 如果失败，回退到原始方法
            pass
    
    # 对于其他情况，使用原始方法
    return _original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

# 应用 monkey patch
AutoTokenizer.from_pretrained = classmethod(patched_from_pretrained)
