import gc
import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache" / "gradio"))

import gradio as gr
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

MODELS = {
    "预设音色": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "设计音色": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "克隆声音": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
}
VOICES = [
    ("Serena · 温柔女声", "Serena"),
    ("Vivian · 明亮女声", "Vivian"),
    ("Uncle Fu · 沉稳男声", "Uncle_Fu"),
    ("Dylan · 北京男声", "Dylan"),
    ("Eric · 四川男声", "Eric"),
]
DEFAULT_STYLE = "成年女性，普通话清晰，温柔自然，语速适中，像专业导航播报员。"
_lock = Lock()
_model = None
_model_id = None


def synthesize(mode, text, voice, instruction, reference, transcript):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if mode not in MODELS:
        raise gr.Error("请选择生成方式。")
    if not lines:
        raise gr.Error("请输入要生成的台词。")
    if len(lines) > 20 or any(len(line) > 200 for line in lines):
        raise gr.Error("每次最多生成 20 句，每句最多 200 字。")
    if mode == "设计音色" and not instruction.strip():
        raise gr.Error("请填写音色描述。")
    if mode == "预设音色" and voice not in dict(VOICES).values():
        raise gr.Error("请选择预设音色。")
    if mode == "克隆声音":
        if not reference or not transcript.strip():
            raise gr.Error("请提供参考录音和与录音一致的文字。")
        info = sf.info(reference)
        if not 3 <= info.duration <= 30:
            raise gr.Error("请使用 3 至 30 秒的清晰参考录音。")

    global _model, _model_id
    with _lock:
        started = perf_counter()
        model_id = MODELS[mode]
        if _model_id != model_id:
            _model = None
            _model_id = None
            gc.collect()
            mx.clear_cache()
            gr.Info("正在加载声音模型，首次使用会下载约 4.5 GB，请保持页面打开。")
            _model = load_model(model_id)
            _model_id = model_id

        folder = ROOT / "outputs" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
        folder.mkdir(parents=True)
        files = []
        for index, line in enumerate(lines, 1):
            options = {"text": line, "lang_code": "Chinese", "max_tokens": 2048}
            if mode == "克隆声音":
                options.update(ref_audio=reference, ref_text=transcript.strip())
            else:
                options["instruct"] = instruction.strip() or None
                if mode == "预设音色":
                    options["voice"] = voice
            chunks = []
            sample_rate = None
            for result in _model.generate(**options):
                chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
                sample_rate = result.sample_rate
            if not chunks or not any(chunk.size for chunk in chunks):
                raise gr.Error("模型未生成音频，请重试或缩短台词。")
            audio = np.concatenate(chunks)
            if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-6:
                raise gr.Error("生成音频异常，请重试。")
            peak = np.max(np.abs(audio))
            if peak > 0.99:
                audio = audio * (0.99 / peak)
            path = folder / f"{index:02d}.wav"
            sf.write(path, audio, sample_rate, subtype="PCM_16")
            files.append(path)

        manifest = folder / "台词.json"
        manifest.write_text(json.dumps({
            "model": model_id, "mode": mode,
            "voice": voice if mode == "预设音色" else None,
            "instruction": instruction if mode != "克隆声音" else None,
            "items": [{"file": path.name, "text": line} for path, line in zip(files, lines)],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        archive = folder / "语音包素材.zip"
        with ZipFile(archive, "w") as bundle:
            for path in [*files, manifest]:
                bundle.write(path, path.name)
        elapsed = perf_counter() - started
        return str(files[0]), [str(path) for path in [*files, archive]], f"已生成 {len(files)} 句，用时 {elapsed:.1f} 秒。保存位置：{folder}"


def build_ui():
    with gr.Blocks(title="本地语音包工作台") as demo:
        gr.Markdown("# 本地语音包工作台\n输入台词，生成中文语音。每行单独导出一个 WAV，可试听并下载整包素材。")
        mode = gr.Dropdown(list(MODELS), value="设计音色", label="生成方式", filterable=False)
        text = gr.Textbox(label="台词（每行一句）", lines=6, value="前方三百米右转，请提前驶入右侧车道。")
        with gr.Group() as style_group:
            voice = gr.Dropdown(VOICES, value="Serena", label="预设音色", visible=False)
            instruction = gr.Textbox(label="音色或表达描述", value=DEFAULT_STYLE, lines=2)
        with gr.Group(visible=False) as clone_group:
            reference = gr.Audio(label="参考录音（3–30 秒）", sources=["upload", "microphone"], type="filepath", format="wav")
            transcript = gr.Textbox(label="参考录音对应文字", lines=3)
        gr.Markdown("首次使用每种方式需下载模型，之后可本地生成。整套台词需要统一音色时，先设计并下载满意的声音，再将它作为克隆参考。导出的 ZIP 是音频素材合集，高德录入需另行操作。")
        generate = gr.Button("生成语音", variant="primary")
        preview = gr.Audio(label="第一句试听", type="filepath")
        downloads = gr.File(label="下载音频 / 全部素材", file_count="multiple")
        status = gr.Textbox(label="生成结果", interactive=False)
        mode.change(
            lambda value: (gr.update(visible=value != "克隆声音"), gr.update(visible=value == "预设音色"), gr.update(visible=value == "克隆声音")),
            mode, [style_group, voice, clone_group], queue=False,
        )
        generate.click(synthesize, [mode, text, voice, instruction, reference, transcript], [preview, downloads, status], concurrency_limit=1)
    return demo


if __name__ == "__main__":
    build_ui().queue(max_size=8).launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
