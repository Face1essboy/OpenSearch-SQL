from modelscope.hub.snapshot_download import snapshot_download
import os

# 1. 定义你想要的特定路径（这里设为项目下的 models/bge-m3）
# 你可以改成任何你喜欢的绝对或相对路径
target_dir = "./models/bge-m3"

print(f"开始下载 BGE-M3 到: {os.path.abspath(target_dir)} ...")

# 2. 从 ModelScope 下载
# 这会下载完整的文件（包括 safetensors），没有符号链接/替身问题
path = snapshot_download(
    'BAAI/bge-m3', # 在 ModelScope 上的 ID，与 HF 内容一致
    local_dir=target_dir
)

print(f"✅ 下载完成！模型已存放在: {path}")