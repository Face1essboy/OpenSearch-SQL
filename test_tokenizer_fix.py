#!/usr/bin/env python3
"""
测试修复 tokenizer 加载问题
"""
import json
from pathlib import Path

# 检查 tokenizer_config.json 是否有问题
tokenizer_config_path = Path("models/bge-m3/tokenizer_config.json")

if tokenizer_config_path.exists():
    with open(tokenizer_config_path, 'r') as f:
        config = json.load(f)
    
    print("当前 tokenizer_config.json 内容:")
    print(json.dumps(config, indent=2))
    
    # 检查是否需要添加 model_type
    if 'model_type' not in config:
        print("\n⚠️  tokenizer_config.json 缺少 model_type 字段")
        print("尝试添加 model_type...")
        
        # 从 config.json 读取 model_type
        config_path = Path("models/bge-m3/config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            if 'model_type' in model_config:
                config['model_type'] = model_config['model_type']
                print(f"添加 model_type: {config['model_type']}")
                
                # 备份原文件
                backup_path = tokenizer_config_path.with_suffix('.json.bak')
                import shutil
                shutil.copy(tokenizer_config_path, backup_path)
                print(f"已备份原文件到: {backup_path}")
                
                # 写入修复后的配置
                with open(tokenizer_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print("✓ 已修复 tokenizer_config.json")
            else:
                print("✗ config.json 中也找不到 model_type")
        else:
            print("✗ 找不到 config.json")
    else:
        print(f"\n✓ tokenizer_config.json 已有 model_type: {config['model_type']}")
else:
    print("✗ 找不到 tokenizer_config.json")

