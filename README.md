<h1 align="center">
<span>Detecting Hallucinations in Authentic LLM-Human Interactions</span>
</h1>
![benchmark_process](assets/benchmark_comstruction_pipeline.png)
**AuthenHallu** is a hallucination detection benchmark entirely grounded
in authentic LLM–human interactions. This repository provides the full dataset and the accompanying experimental code.

[📝 Full Paper](https://arxiv.org/abs/2510.10539) | [🤗 Dataset](https://huggingface.co/datasets/Yujie-AI/AuthenHallu)


## Benchmark Overview
### Basic Statistics
| Key   | Value |
|-------|:-------:|
|# Dialogues | 400 |
|# Hallucinated dialogues | 163 |
|# Query–response pairs per dialogue | 2 |
|# Total query–response pairs | 800 |
|# Hallucinated query–response pairs | 251 |
|# Tokens per query (avg.) | 20 |
|# Tokens per response (avg.) | 134 |

### Data Format
Each entry corresponds to a dialogue with two query-response pairs and provides pair-level hallucination annotations, including both binary occurrence labels and fine-grained category labels.  Our dataset is constructed based on [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m). To comply with the [LMSYS-Chat-1M Dataset License Agreement](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), we do not redistribute the original dialogue content. Instead, users can retrieve the corresponding dialogues from the source dataset using the provided `conversation_id`.
| Field | Type | Description |
|-------|------|-------------|
| `conversation_id` | string | A unique identifier for each dialogue, corresponding to the `conversation_id` in the LMSYS-Chat-1M dataset. |
| `occurrence1` | string | Binary hallucination occurrence label for the first query-response pair, selected from {`Hallucination`, `No Hallucination`}. |
| `category1` | string | Hallucination type for the first query-response pair, selected from {`Input-conflicting`, `Context-conflicting`, `Fact-conflicting`, `None`}. |
| `occurrence2` | string | Binary hallucination occurrence label for the second query-response pair, selected from {`Hallucination`, `No Hallucination`}. |
| `category2` | string | Hallucination type for the second query-response pair, selected from {`Input-conflicting`, `Context-conflicting`, `Fact-conflicting`, `None`}. |

## Getting Started
### Prerequisites
- Python 3.11
- You typically need to reconstruct the full dataset yourself using the original LMSYS-Chat-1M dialogue content. In accordance with the LMSYS-Chat-1M Dataset License Agreement, we are unable to redistribute the original dialogue data in this repository.

### Installing dependencies
```
pip install -r requirements.txt
```
### Using the benchmark
You can access the AuthenHallu benchmark through the `AuthenHallu.json` file in this repository or via the [AuthenHallu](https://huggingface.co/datasets/Yujie-AI/AuthenHallu) repository on Hugging Face.
```
from datasets import load_dataset

# Load dataset from this repository
ds = load_dataset(path="json", data_files="AuthenHallu.json", split="train")

# Iterate through each example
for example in ds:
    conversation_id = example.get("conversation_id")
    occurrence1 = example.get("occurrence1")
    category1 = example.get("category1")
    occurrence2 = example.get("occurrence2")
    category2 = example.get("category2")
```


## Citation
```bibtex
@article{ren2025detecting,
  title={Detecting Hallucinations in Authentic LLM-Human Interactions},
  author={Ren, Yujie and Gruhlke, Niklas and Lauscher, Anne},
  journal={arXiv preprint arXiv:2510.10539},
  year={2025}
}
```