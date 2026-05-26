"""
Qwen3-TTS Base 模型，用于语音克隆。

当参考音频与目标音频不是同一语种时，建议开启零样本克隆模式（zero_shot=True），以获得更好的音色迁移效果。

开启零样本克隆后，构建 prompt 时将不再使用参考文本和编码，而仅使用从参考音频提取的说话人嵌入（spk_embd）作为锚点。
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
    os.makedirs("./output/design", exist_ok=True)
    
    # 设置音色锚点

    # 读取音频文件，需要编码为 Code，是有损克隆
    # REF_AUDIO = "output/elaborate/Vivian.wav"                
    # REF_TEXT = "你好，我是千问，你今天过得好吗？"
    # stream.set_voice(REF_AUDIO, REF_TEXT)

    # 从 json 读取 code，无需从 wav 编码，可以无损克隆
    REF_JSON = "output/elaborate/Vivian.json"           
    stream.set_voice(REF_JSON)
    
    
    # if stream.voice:
    #     print(f"🔊 正在还原并播放参考音频: {REF_JSON} ...")
    #     engine.decode(stream.voice)
    #     engine.encode(stream.voice)
    #     stream.voice.play(blocking=True)
    

    print(f"\n🎙️  [2/2] 开始流式推理 (边推边播)...")
    target_text = "Adding 8 frames of zero-pressure preheating inference during the construction phase will force the inference engine (such as DML) to complete the allocation of the computation graph and memory optimization in advance, thereby ensuring that the first formal inference is in the optimal peak state."
    config = TTSConfig(
        max_steps=400, 
        temperature=0.6, 
        sub_temperature=0.6, 
        seed=42, 
        sub_seed=45,
        streaming=True,
    )
    # config = TTSConfig(max_steps=400, do_sample=False, sub_do_sample=False)
    result = stream.clone(
        text=target_text, 
        language='English', 
        zero_shot=True,
        config=config, 
    )
    result.print_stats()

    text_prefix = "".join(c for c in target_text.strip() if c not in r'<>:"/\|?*')[:20].strip()
    save_path = f"./output/clone/{text_prefix}"
    result.save(f"{save_path}.wav")     # 保存为音频
    result.save(f"{save_path}.json")    # 保存为json，内含无损的音频code

    stream.join()
    
    engine.shutdown()

if __name__ == "__main__":
    main()
