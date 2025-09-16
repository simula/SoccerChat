# SoccerChat

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
[![Model on HF](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/SimulaMet/SoccerChat-qwen2-vl-7b)
[![Dataset on HF](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/SimulaMet/SoccerChat)
[![arXiv](https://img.shields.io/badge/arXiv-2505.16630-b31b1b.svg)](https://arxiv.org/abs/2505.16630)

*Multimodal soccer game understanding: model & dataset*

---

## Overview

**SoccerChat** is centered around integrating video, event annotations, and commentary text to support advanced soccer game understanding. The project includes:

* A **vision-language model**, *SoccerChat-qwen2-vl-7b*, fine-tuned for soccer video + text tasks.
* A **multimodal dataset**, *SoccerChat*, pairing soccer video clips with questions, responses, and event labels.
* A paper: *“SoccerChat: Integrating Multimodal Data for Enhanced Soccer Game Understanding”* (arXiv: **2505.16630**).

The GitHub repository is home to all code for training, evaluation, and usage, including dataset processing.

---

## Resources

| Asset   | Description                                                                                                                         | Link                                                                                                  |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Model   | SoccerChat-qwen2-vl-7b — A LoRA-finetuned version of *Qwen2-VL-7B-Instruct*, optimized for soccer video understanding and dialogue. | [HuggingFace Model Card](https://huggingface.co/SimulaMet/SoccerChat-qwen2-vl-7b/blob/main/README.md) |
| Dataset | SoccerChat — \~90k multimodal examples, each with video, natural-language query & response, and event labels.                       | [HuggingFace Dataset Card](https://huggingface.co/datasets/SimulaMet/SoccerChat/blob/main/README.md)  |
| Paper   | ArXiv preprint describing the model, dataset, experiments, and findings.                                                            | [arXiv: 2505.16630](https://arxiv.org/abs/2505.16630)                                                 |

---

## Dataset: SoccerChat

* **Size & Splits**

  * **Train**: \~85,220 examples
  * **Validation**: \~4,625 examples

* **Features per example**

  * `video`: short previewable soccer clip
  * `query`: a natural‐language question about the video
  * `response`: natural‐language answer
  * `events`: zero or more event types (using SoccerNet event taxonomy)
  * `path`: relative path to the video file

* **Modality & Tasks**

  * Modalities: video + text
  * Supporting tasks include video classification and video‐text‐to‐text (QA, commentary generation, event detection)

* **Storage & Download**

  * Videos are large (\~48 GB for full video set) and stored via Git-LFS.
  * Full dataset files are in parquet format.

* **Sample usage**

  ```python
  from datasets import load_dataset

  ds = load_dataset("SimulaMet/SoccerChat")

  # Optionally convert to JSONL for use with MS-SWIFT or similar
  for split in ["train", "validation"]:
      df = ds[split].to_pandas()
      df["query"] = "<video>" + df["query"]
      df["videos"] = df["path"].apply(lambda p: [os.path.join(base, os.path.basename(p))])
      df[["query", "response", "videos"]].to_json(f"{split}.jsonl", orient="records", lines=True)
  ```

---

## Model: SoccerChat-qwen2-vl-7b

* **What it is**
  A **LoRA-finetuned** version of *Qwen2-VL-7B-Instruct*, trained on the SoccerChat dataset. Tailored for interpreting and reasoning over multimodal soccer content – combining video frames, commentary, and annotated events.

* **Capabilities**

  * Answering questions about match videos (e.g. “What happened?”, “Who scored?”, “Why was play stopped?”)
  * Generating match commentary aligned to the video + event context
  * Event‐based reasoning (goals, fouls, substitutions, etc.)

* **Limitations & Scope**

  * Domain: Soccer domain only; limited or no generalization outside soccer.
  * Language: English only; datasets & commentary are English.
  * Risks: Possible hallucinations in ambiguous video segments.

* **Getting Started — Example Code Snippet**

  * Use the code below to get started with the model.
  The model accepts **video + text queries**.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Simula/SoccerChat/blob/main/notebooks/usage.ipynb)

  ```python
  import os
  import torch
  from swift.llm import PtEngine, RequestConfig, InferRequest
  from transformers import BitsAndBytesConfig

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.float16
  )

  os.environ["FPS_MIN_FRAMES"] = "24"
  os.environ["FPS_MAX_FRAMES"] = "24"
  os.environ["VIDEO_MAX_PIXELS"] = "100352"

  engine = PtEngine(
      adapters=["SimulaMet/SoccerChat-qwen2-vl-7b"],
      quantization_config=bnb_config,
      attn_impl="sdpa",
      max_batch_size=1,
      use_hf=True,
      model_id_or_path="Qwen/Qwen2-VL-7B-Instruct",
  )

  req_cfg = RequestConfig(max_tokens=512, temperature=0.3, top_k=20, top_p=0.7, repetition_penalty=1.05)

  infer_requests = [
      InferRequest(messages=[{
          "role": "user",
          "content": [
              {"type": "video", "video": "https://huggingface.co/datasets/SimulaMet/SoccerChat/resolve/main/videos/MultipleEvents/100037_Shotsontarget--Balloutofplay.mp4"},
              {"type": "text", "text": "What is shown in the video?"}
          ],
      }])
  ]

  resp = engine.infer(infer_requests, req_cfg)
  print(resp[0].choices[0].message.content)
  ```

---

## How to Train & Evaluate

* **Dataset conversion**: convert the Hugging Face dataset into JSONL format for models/frameworks that consume `{"query", "response", "videos"}` examples.

* **Training** (example)

  ```bash
  NFRAMES=24 MAX_PIXELS=100352 NPROC_PER_NODE=4 swift sft \
    --model_type qwen2-vl-7b-instruct \
    --model_id_or_path qwen/Qwen2-VL-7B-Instruct \
    --sft_type lora \
    --dataset SoccerChat+XFoul_train.jsonl \
    --num_train_epochs 5 \
    --batch_size 14 \
    --deepspeed default-zero2 \
    --eval_steps 100 \
    --dataset_test_ratio 0.05
  ```

* **Evaluation / Inference**

  ```bash
  NFRAMES=24 MAX_PIXELS=100352 swift infer \
    --ckpt_dir checkpoint-dir \
    --load_dataset_config true \
    --merge_lora true \
    --val_dataset XFoul_valid.jsonl
  ```

---

## Training Details & Evaluation

* **Training setup**

  * Base model: *Qwen2-VL-7B-Instruct*
  * Finetuning with LoRA (via PEFT)
  * Mixed precision (fp16) training

* **Evaluation metrics**

  * Text generation metrics: BLEU, ROUGE, METEOR etc. for commentary and responses.
  * Event detection metrics: accuracy / recall over key match events.
  * Human evaluation for fluency and correctness (see paper).

* **Model performance**

  * In the paper, the model shows improved performance over baseline VLMs on tasks such as event detection, commentary generation, and QA in soccer domain.

---

## Citation

If you use either the **SoccerChat dataset** or the **SoccerChat-qwen2-vl-7b model**, please cite:

```bibtex
@article{Gautam2025May,
  author = {Gautam, Sushant and Midoglu, Cise and Thambawita, Vajira and Riegler, Michael A. and Halvorsen, Pål and Shah, Mubarak},
  title = {SoccerChat: Integrating Multimodal Data for Enhanced Soccer Game Understanding},
  journal = {ArXiv e-prints},
  year = {2025},
  month = may,
  eprint = {2505.16630},
  doi = {10.48550/arXiv.2505.16630}
}
```

---

## Contact & License

* **Organization:** SimulaMet
* **Issues / Feedback:** via GitHub issues in this repo
* **License:** Apache-2.0 (for model); dataset is shared under terms specified on the Hugging Face dataset card.
