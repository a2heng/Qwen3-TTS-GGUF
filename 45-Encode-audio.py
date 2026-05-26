"""
将 wav 音频编码，保存到 json 中，内含 spk_embd 与 codes，供后续克隆或 custom voice使用。

注意，官方提供的模型中，只有 base 模型中能提取到 speaker encoder，custom 模型没有编码器。当然，也可以把提取到的编码器放到 custom voice 模型中使用。
"""
import time
import os
import numpy as np
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult

# ==================== Vulkan 选项 ====================

# os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
# os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"   # 禁止 Vulkan 用独显（强制用集显）
# os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）


def main():

    # 初始化引擎
    print("🚀 [Base-Clone] 正在初始化 TTS 引擎...")
    engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA")
    stream = engine.create_stream()

    # 确保输出目录存在
    os.makedirs("./output", exist_ok=True)
    
    # 编码音频文件，提取 voice，即 spk_embd 与 codes
    REF_AUDIO = "output/elaborate/Vivian.wav"                
    REF_TEXT = "你好，我是千问，你今天过得好吗？"
    stream.set_voice(REF_AUDIO, REF_TEXT)

    # 保存为 json，内含 spk_embd 与 codes
    stream.voice.save("output/voice.json")
    
    engine.shutdown()

if __name__ == "__main__":
    main()
