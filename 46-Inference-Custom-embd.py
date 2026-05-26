"""
从 json 读取 spk_embd 作为说话人锚点，而不使用内置说话人的 spk_embd
"""
import time
import os
import numpy as np
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult



def main():
    
    # 1. 初始化引擎
    print(f"🚀 [Custom-Inference] 正在初始化引擎")
    engine = TTSEngine(model_dir="model-custom", onnx_provider='CUDA')
    stream = engine.create_stream()
    
    # 确保输出目录存在
    os.makedirs("./output/custom", exist_ok=True)

    config = TTSConfig(
        max_steps=400, 
        temperature=0.6, 
        sub_temperature=0.6, 
        seed=42, 
        sub_seed=45,
        streaming=True,
    )

    voice = TTSResult.from_json("output/voice.json")

    result = stream.custom(
        text = '使用从音频提取的说话人嵌入，进行 custom voice 生成。', 
        speaker=voice.spk_emb,
        instruct = '以极度悲伤、带着明显哭腔的语气，用较小的音量缓缓诉说，语速缓慢',
        language = "Chinese",
        config=config, 
    )

    save_path = "./output/zero-shot-clone"
    result.save(f"{save_path}.wav")
    result.save(f"{save_path}.json")

    print(f"💾 已保存至: {save_path}.wav / .json")

    result.print_stats()
    stream.join()

    engine.shutdown()

if __name__ == "__main__":
    main()
