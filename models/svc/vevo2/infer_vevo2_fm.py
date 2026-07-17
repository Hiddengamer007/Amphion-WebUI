import os
import torch
from huggingface_hub import snapshot_download
import argparse
from models.svc.vevo2.vevo2_utils import *


def vevo2_fm(content_wav_path, reference_wav_path, output_path, shifted_src=True):
    gen_audio = inference_pipeline.inference_fm(
        src_wav_path=content_wav_path,
        timbre_ref_wav_path=reference_wav_path,
        use_pitch_shift=shifted_src,
        flow_matching_steps=32,
    )
    save_audio(gen_audio, output_path=output_path)


def load_inference_pipeline():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    local_dir = snapshot_download(
        repo_id="RMSnow/Vevo2",
        repo_type="model",
        local_dir="./ckpts/Vevo2",
        resume_download=True,
    )

    content_style_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
    )

    fmt_cfg_path = os.path.join(
        local_dir, "acoustic_modeling/fm_emilia101k_singnet7k_repa/config.json"
    )
    fmt_ckpt_path = os.path.join(
        local_dir, "acoustic_modeling/fm_emilia101k_singnet7k_repa"
    )

    vocoder_cfg_path = os.path.join(local_dir, "vocoder/config.json")
    vocoder_ckpt_path = os.path.join(local_dir, "vocoder")

    inference_pipeline = Vevo2InferencePipeline(
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )
    return inference_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vevo2 SVC voice conversion.")
    
    parser.add_argument(
        "--content", "-c", 
        required=True, 
        help="Path to the vocal file to be converted"
    )
    parser.add_argument(
        "--reference", "-r", 
        required=True, 
        help="Path to the reference voice file"
    )
    parser.add_argument(
        "--output_dir", "-o", 
        default="./models/svc/vevo2/output", 
        help="Directory where the output will be saved (default: ./models/svc/vevo2/output)"
    )

    args = parser.parse_args()

    inference_pipeline = load_inference_pipeline()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "svc.wav")

    vevo2_fm(args.content, args.reference, output_path)

    vevo2_fm(content_wav_path, reference_wav_path, output_path)
