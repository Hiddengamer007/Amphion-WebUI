from pyngrok import ngrok

"""
Gradio WebUI for Vevo2 (Amphion), replicating:

    python -m models.svc.vevo2.infer_vevo2_fm

Run from the Amphion repo root:

    python vevo2_webui.py

It wraps `Vevo2InferencePipeline` from models/svc/vevo2/vevo2_utils.py and exposes
the FM (voice/singing conversion) flow that the command runs, plus the AR+FM tasks
(TTS, editing, singing style conversion, melody control).

Resource notes for weak hardware:
  * Use device = "cpu" (the default is auto -> cuda if available).
  * Lower "Flow-matching steps" (e.g. 16) to cut compute/time.
  * Enable chunking for long inputs to bound peak memory.
  * Use the "Unload model" button to free RAM/VRAM when not generating.
  
New features:
  * Multithreaded chunked audio processing for faster FM inference on long files
  * Dual GPU support (Kaggle 2x NVIDIA T4) with GPU selection in UI
"""

import os
import sys
import argparse
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Make sure the Amphion repo root (this file's folder) is importable so that
# `from models.svc.vevo2.vevo2_utils import ...` resolves, exactly like `python -m`.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gradio as gr  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from models.svc.vevo2.vevo2_utils import (  # noqa: E402
    Vevo2InferencePipeline,
    save_audio,
)

MODEL_REPO_ID = "RMSnow/Vevo2"
CKPT_DIR = os.path.join(REPO_ROOT, "ckpts", "Vevo2")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global, single pipeline instance + a lock so we never run two inferences at once.
PIPELINE = None
PIPELINE_MODE = None
LOCK = threading.Lock()

# GPU configuration
GPU_DEVICES = []
GPU_PIPELINES = {}


def _ckpt(*parts):
    return os.path.join(CKPT_DIR, *parts)


def get_available_gpus():
    """Detect available GPUs (supporting Kaggle 2x T4 setup)."""
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def load_pipeline(device, mode, cpu_threads=0, use_multi_gpu=False):
    """Download (if needed) and build the Vevo2 inference pipeline.

    Mirrors load_inference_pipeline() in both infer_vevo2_fm.py and infer_vevo2_ar.py.
    
    Args:
        device: Primary device string ("auto", "cpu", "cuda:0", "cuda:1", etc.)
        mode: "FM only" or "AR+FM"
        cpu_threads: Number of CPU threads (0 = auto)
        use_multi_gpu: If True, load pipelines on ALL available GPUs for parallel chunk processing
    """
    global PIPELINE, PIPELINE_MODE, GPU_DEVICES, GPU_PIPELINES

    # Detect GPU devices
    available_gpus = get_available_gpus()
    
    if device == "auto":
        if available_gpus:
            dev = torch.device(available_gpus[0])
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    if cpu_threads and cpu_threads > 0:
        torch.set_num_threads(int(cpu_threads))

    local_dir = snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        local_dir=CKPT_DIR,
        resume_download=True,
    )

    # Paths shared by FM and AR pipelines.
    content_style_tokenizer_ckpt_path = _ckpt("tokenizer", "contentstyle_fvq16384_12.5hz")
    fmt_cfg_path = _ckpt("acoustic_modeling", "fm_emilia101k_singnet7k_repa", "config.json")
    fmt_ckpt_path = _ckpt("acoustic_modeling", "fm_emilia101k_singnet7k_repa")
    vocoder_cfg_path = _ckpt("vocoder", "config.json")
    vocoder_ckpt_path = _ckpt("vocoder")

    def _build_pipeline(device_str):
        """Build a pipeline on a specific device."""
        dev_obj = torch.device(device_str)
        
        if mode == "AR+FM":
            prosody_tokenizer_ckpt_path = _ckpt("tokenizer", "prosody_fvq512_6.25hz")
            ar_cfg_path = _ckpt("contentstyle_modeling", "posttrained", "amphion_config.json")
            ar_ckpt_path = _ckpt("contentstyle_modeling", "posttrained")

            return Vevo2InferencePipeline(
                prosody_tokenizer_ckpt_path=prosody_tokenizer_ckpt_path,
                content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
                ar_cfg_path=ar_cfg_path,
                ar_ckpt_path=ar_ckpt_path,
                fmt_cfg_path=fmt_cfg_path,
                fmt_ckpt_path=fmt_ckpt_path,
                vocoder_cfg_path=vocoder_cfg_path,
                vocoder_ckpt_path=vocoder_ckpt_path,
                device=dev_obj,
            )
        else:  # FM only
            return Vevo2InferencePipeline(
                content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
                fmt_cfg_path=fmt_cfg_path,
                fmt_ckpt_path=fmt_ckpt_path,
                vocoder_cfg_path=vocoder_cfg_path,
                vocoder_ckpt_path=vocoder_ckpt_path,
                device=dev_obj,
            )

    # Load primary pipeline
    PIPELINE = _build_pipeline(str(dev))
    PIPELINE_MODE = mode
    
    # Load on all GPUs if multi-GPU is enabled
    GPU_DEVICES = []
    GPU_PIPELINES = {}
    
    if use_multi_gpu and len(available_gpus) > 1:
        for gpu_id in available_gpus:
            try:
                GPU_PIPELINES[gpu_id] = _build_pipeline(gpu_id)
                GPU_DEVICES.append(gpu_id)
            except Exception as e:
                print(f"Warning: Failed to load pipeline on {gpu_id}: {e}")
        
        if GPU_DEVICES:
            return f"Loaded '{mode}' pipeline on {len(GPU_DEVICES)} GPUs: {', '.join(GPU_DEVICES)}."
    
    return f"Loaded '{mode}' pipeline on {dev}."


def unload_pipeline():
    global PIPELINE, PIPELINE_MODE, GPU_PIPELINES, GPU_DEVICES
    PIPELINE = None
    PIPELINE_MODE = None
    GPU_PIPELINES = {}
    GPU_DEVICES = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    return "Pipeline unloaded; memory released."


def _crossfade_concat(parts, sr, cf_sec=0.5):
    cf = int(cf_sec * sr)
    if len(parts) == 1:
        return parts[0]
    out = parts[0]
    for p in parts[1:]:
        if cf <= 0 or len(out) < cf or len(p) < cf:
            out = np.concatenate([out, p])
            continue
        fade_out = np.linspace(1.0, 0.0, cf)
        fade_in = np.linspace(0.0, 1.0, cf)
        mixed = out[-cf:] * fade_out + p[:cf] * fade_in
        out = np.concatenate([out[:-cf], mixed, p[cf:]])
    return out


def _save(tensor_audio, out_name):
    if not out_name.endswith(".wav"):
        out_name += ".wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    save_audio(tensor_audio, sr=24000, output_path=out_path)
    return out_path


def _process_single_chunk(args):
    """Process a single audio chunk (called by ThreadPoolExecutor)."""
    seg_path, reference, pitch_shift, steps, gpu_id = args
    
    # Use GPU-specific pipeline if available
    if gpu_id and gpu_id in GPU_PIPELINES:
        pipeline = GPU_PIPELINES[gpu_id]
    else:
        pipeline = PIPELINE
    
    audio = pipeline.inference_fm(
        src_wav_path=seg_path,
        timbre_ref_wav_path=reference,
        use_pitch_shift=bool(pitch_shift),
        flow_matching_steps=steps,
    )
    return audio.squeeze(0).numpy()


def run_fm(source, reference, steps, pitch_shift, chunk, chunk_dur, out_name, 
           use_multi_gpu=False, progress=gr.Progress()):
    """Replicates vevo2_fm(): source = content/prosody, reference = target timbre.
    
    Now with multithreaded chunk processing and multi-GPU support.
    """
    if PIPELINE is None:
        return None, "Load the model first (use the Model tab)."
    if source is None or reference is None:
        return None, "Provide both a source (content) and a reference (timbre) audio."

    steps = int(steps)
    out_name = (out_name or "svc").strip() or "svc"

    with LOCK:
        try:
            use_chunk = bool(chunk) and chunk_dur and float(chunk_dur) > 0
            if use_chunk:
                import librosa
                import torchaudio

                y, _ = librosa.load(source, sr=24000)
                chunk_samples = int(float(chunk_dur) * 24000)
                if len(y) / 24000 > float(chunk_dur):
                    progress(0.0, desc="Chunking source")
                    bounds = list(range(0, len(y), chunk_samples))
                    parts = []
                    tmp = tempfile.mkdtemp()
                    n = len(bounds)
                    
                    # Prepare chunk files
                    chunk_files = []
                    for i, start in enumerate(bounds):
                        seg = y[start : start + chunk_samples]
                        seg_path = os.path.join(tmp, f"seg_{i}.wav")
                        torchaudio.save(seg_path, torch.from_numpy(seg).unsqueeze(0), 24000)
                        chunk_files.append(seg_path)
                    
                    # Determine GPU assignment for each chunk
                    gpu_pool = GPU_DEVICES if (use_multi_gpu and GPU_DEVICES) else [None]
                    
                    # Build task list with GPU round-robin assignment
                    tasks = []
                    for i, seg_path in enumerate(chunk_files):
                        gpu_id = gpu_pool[i % len(gpu_pool)]
                        tasks.append((seg_path, reference, pitch_shift, steps, gpu_id))
                    
                    # Process chunks in parallel using ThreadPoolExecutor
                    progress(0.1, desc=f"Processing {n} chunks on {len(gpu_pool)} GPU(s)...")
                    
                    results = [None] * n
                    completed = 0
                    
                    with ThreadPoolExecutor(max_workers=len(gpu_pool)) as executor:
                        future_to_idx = {
                            executor.submit(_process_single_chunk, task): i 
                            for i, task in enumerate(tasks)
                        }
                        
                        for future in as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            try:
                                results[idx] = future.result()
                                completed += 1
                                progress(
                                    0.1 + 0.8 * (completed / n), 
                                    desc=f"FM chunk {completed}/{n} done"
                                )
                            except Exception as e:
                                return None, f"Error processing chunk {idx}: {e}"
                    
                    # Concatenate results in order
                    audio = torch.from_numpy(_crossfade_concat(results, 24000))
                    out_path = _save(audio.unsqueeze(0), out_name)
                    gpu_info = f" ({len(gpu_pool)} GPU(s) parallel)" if len(gpu_pool) > 1 else ""
                    return out_path, f"Done (chunked, {n} segments{gpu_info})."
                # falls through to normal path if shorter than one chunk

            progress(0.1, desc="Running FM inference")
            audio = PIPELINE.inference_fm(
                src_wav_path=source,
                timbre_ref_wav_path=reference,
                use_pitch_shift=bool(pitch_shift),
                flow_matching_steps=steps,
            )
            out_path = _save(audio, out_name)
            return out_path, "Done."
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"


def run_ar(task, target_text, raw_audio, raw_text, style_audio, style_text,
           timbre_audio, melody_audio, steps, out_name, progress=gr.Progress()):
    """Replicates the functions in infer_vevo2_ar.py (TTS / Editing / Style / Melody)."""
    if PIPELINE is None:
        return None, "Load the model first (use the Model tab)."
    if getattr(PIPELINE, "ar_model", None) is None:
        return None, "Pipeline is FM-only. Reload in 'AR+FM' mode for these tasks."
    steps = int(steps)
    out_name = (out_name or "ar_out").strip() or "ar_out"

    with LOCK:
        try:
            progress(0.1, desc=f"Running {task}")
            if task == "TTS":
                timbre = timbre_audio if timbre_audio else style_audio
                audio = PIPELINE.inference_ar_and_fm(
                    target_text=target_text or "",
                    style_ref_wav_path=style_audio,
                    style_ref_wav_text=raw_text or "",
                    timbre_ref_wav_path=timbre,
                    use_prosody_code=False,
                    flow_matching_steps=steps,
                )
            elif task == "Editing":
                audio = PIPELINE.inference_ar_and_fm(
                    target_text=target_text or "",
                    prosody_wav_path=raw_audio,
                    style_ref_wav_path=raw_audio,
                    style_ref_wav_text=raw_text or "",
                    timbre_ref_wav_path=raw_audio,
                    use_prosody_code=True,
                    flow_matching_steps=steps,
                )
            elif task == "Singing Style Conversion":
                audio = PIPELINE.inference_ar_and_fm(
                    target_text=raw_text or "",
                    prosody_wav_path=raw_audio,
                    style_ref_wav_path=style_audio,
                    style_ref_wav_text=style_text or "",
                    timbre_ref_wav_path=raw_audio,
                    use_prosody_code=True,
                    use_pitch_shift=True,
                    flow_matching_steps=steps,
                )
            elif task == "Melody Control":
                timbre = timbre_audio if timbre_audio else style_audio
                audio = PIPELINE.inference_ar_and_fm(
                    target_text=target_text or "",
                    prosody_wav_path=melody_audio,
                    style_ref_wav_path=style_audio,
                    style_ref_wav_text=style_text or "",
                    timbre_ref_wav_path=timbre,
                    use_prosody_code=True,
                    use_pitch_shift=True,
                    flow_matching_steps=steps,
                )
            else:
                return None, f"Unknown task: {task}"

            out_path = _save(audio, out_name)
            return out_path, f"Done ({task})."
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"


def build_ui():
    with gr.Blocks(title="Vevo2 WebUI") as demo:
        gr.Markdown(
            "# Vevo2 WebUI (Amphion)\\n"
            "Mirror of `python -m models.svc.vevo2.infer_vevo2_fm` with extra AR tasks.\\n"
            "**Tip for weak hardware:** device = `cpu`, lower flow-matching steps, enable chunking.\\n"
            "**New:** Multithreaded chunk processing + Dual GPU support (Kaggle 2x T4)."
        )

        with gr.Tab("Model"):
            # Auto-detect available GPUs for dropdown
            gpu_choices = ["auto", "cpu", "mps"]
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_choices.append(f"cuda:{i}")
            
            device = gr.Dropdown(
                gpu_choices,
                value="auto",
                label="Device",
            )
            mode = gr.Radio(
                ["FM only", "AR+FM"],
                value="FM only",
                label="Pipeline mode (FM only = the `infer_vevo2_fm` command)",
            )
            cpu_threads = gr.Number(value=0, precision=0, label="CPU threads (0 = auto)")
            
            # Multi-GPU support toggle
            multi_gpu = gr.Checkbox(
                value=False, 
                label="Use ALL GPUs (Kaggle 2x T4)",
                info="Load pipeline on all available GPUs for parallel chunk processing"
            )
            
            # GPU info display
            gpu_info = gr.Markdown(
                f"**Detected GPUs:** {torch.cuda.device_count()} available" 
                if torch.cuda.is_available() 
                else "**No CUDA GPUs detected**"
            )
            
            load_btn = gr.Button("Load / Download model", variant="primary")
            unload_btn = gr.Button("Unload model (free memory)")
            model_status = gr.Textbox(label="Status", interactive=False)

            load_btn.click(
                lambda d, m, t, mg: load_pipeline(d, m, int(t or 0), mg),
                [device, mode, cpu_threads, multi_gpu],
                model_status,
            )
            unload_btn.click(unload_pipeline, [], model_status)

        with gr.Tab("Voice / Singing Conversion (FM)"):
            gr.Markdown(
                "Source = content & prosody (speech, singing, or even an instrument). "
                "Reference = target voice/timbre. This is exactly what `infer_vevo2_fm` does.\\n"
                "**Chunked mode now uses multithreading** — chunks are processed in parallel across GPUs."
            )
            with gr.Row():
                fm_source = gr.Audio(label="Source (content/prosody)", type="filepath")
                fm_ref = gr.Audio(label="Reference (timbre)", type="filepath")
            with gr.Row():
                fm_steps = gr.Slider(1, 50, value=32, step=1, label="Flow-matching steps")
                fm_pitch = gr.Checkbox(value=True, label="Pitch shift (match timbre pitch region)")
            with gr.Row():
                fm_chunk = gr.Checkbox(value=False, label="Chunk long audio")
                fm_chunk_dur = gr.Number(value=15, label="Chunk duration (s)")
                fm_multi_gpu = gr.Checkbox(
                    value=False, 
                    label="Use Multi-GPU for chunks",
                    info="Process chunks in parallel across all loaded GPUs"
                )
                fm_out = gr.Textbox(value="svc", label="Output file name")
            fm_run = gr.Button("Convert", variant="primary")
            fm_audio = gr.Audio(label="Output", type="filepath")
            fm_status = gr.Textbox(label="Status", interactive=False)
            fm_run.click(
                run_fm,
                [fm_source, fm_ref, fm_steps, fm_pitch, fm_chunk, fm_chunk_dur, fm_out, fm_multi_gpu],
                [fm_audio, fm_status],
            )

        with gr.Tab("TTS / Editing / Style / Melody (AR+FM)"):
            gr.Markdown("Requires the pipeline loaded in **AR+FM** mode.")
            ar_task = gr.Radio(
                ["TTS", "Editing", "Singing Style Conversion", "Melody Control"],
                value="TTS",
                label="Task",
            )
            ar_target_text = gr.Textbox(label="Target text", lines=2)
            ar_raw = gr.Audio(label="Raw / source audio (Editing, Style Conversion)", type="filepath")
            ar_raw_text = gr.Textbox(label="Raw / source text (Editing, Style Conversion, Melody ref text)")
            ar_style = gr.Audio(label="Style reference audio (TTS, Style, Melody)", type="filepath")
            ar_style_text = gr.Textbox(label="Style reference text (Style, Melody)")
            ar_timbre = gr.Audio(label="Timbre reference audio (TTS, Melody; optional)", type="filepath")
            ar_melody = gr.Audio(label="Melody audio (Melody Control; humming/piano)", type="filepath")
            with gr.Row():
                ar_steps = gr.Slider(1, 50, value=32, step=1, label="Flow-matching steps")
                ar_out = gr.Textbox(value="ar_out", label="Output file name")
            ar_run = gr.Button("Generate", variant="primary")
            ar_audio = gr.Audio(label="Output", type="filepath")
            ar_status = gr.Textbox(label="Status", interactive=False)
            ar_run.click(
                run_ar,
                [
                    ar_task, ar_target_text, ar_raw, ar_raw_text, ar_style,
                    ar_style_text, ar_timbre, ar_melody, ar_steps, ar_out,
                ],
                [ar_audio, ar_status],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Vevo2 Gradio WebUI")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--mode", default="FM only", help="FM only|AR+FM")
    parser.add_argument("--share", action="store_true", help="Create a Gradio public share link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--multi-gpu", action="store_true", help="Load on all available GPUs")
    args = parser.parse_args()

    # Pre-load if requested
    if args.multi_gpu:
        print("Pre-loading pipeline on all GPUs...")
        load_pipeline(args.device, args.mode, use_multi_gpu=True)

    demo = build_ui()
    
    # NGROK
    #public_url = ngrok.connect(7860)
    #print(f" * Public URL: {public_url}")
    #demo.launch(server_name=args.server_name, server_port=args.port, share=args.share, inbrowser=True)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
if __name__ == "__main__":
    main()
