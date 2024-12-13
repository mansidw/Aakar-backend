LoftQ (LoRA-Fine-Tuning-aware Quantization) is a novel quantization framework designed for large language models (LLMs) that require both quantization and LoRA fine-tuning. It addresses the performance gap observed when quantization and LoRA fine-tuning are applied together on pre-trained models. LoftQ simultaneously quantizes an LLM and finds a suitable low-rank initialization for LoRA fine-tuning, which helps to reduce the discrepancy between quantized and full-precision models, thereby improving generalization in downstream tasks.

The framework integrates low-rank approximation with quantization to jointly approximate the original high-precision pre-trained weights. This approach provides a better initialization point for LoRA fine-tuning, leading to improved performance in downstream tasks such as natural language understanding, question answering, summarization, and natural language generation.

LoftQ has been shown to outperform existing quantization methods, particularly in challenging low-bit precision regimes like 2-bit and 2/4-bit mixed precision. It achieves significant gains in performance metrics such as ROUGE scores for summarization tasks and accuracy for question answering tasks, compared to methods like QLoRA.

LoftQ's method involves alternating between quantization and singular value decomposition (SVD) to approximate the original pre-trained weights. This process helps to mitigate the quantization discrepancy and provides a robust initialization for LoRA fine-tuning. The computational cost of LoftQ is minimal, as it is applied to individual weight matrices and can be executed in parallel.

In experiments, LoftQ consistently outperforms QLoRA across all precision levels. For example, with 4-bit quantization, LoftQ achieves higher ROUGE scores on XSum and CNN/DailyMail datasets compared to QLoRA. It also demonstrates robustness in low-bit scenarios, achieving significant gains in tasks like MNLI and SQuADv1.1 with 2-bit quantization.



## QLoRA Performance with Different Bits

| Number of Bits | Log of Perplexity (Pre-trained LLAMA-2-13b on WikiText-2) |
|---|---|
| 16 | 2.44 |
| 8 | 2.01 |
| 4 | 2.23 |
| 3 | 2.83 |
| 2.5 | 11.37 |
| 2.25 | 11.48 |
| 2 | 11.36 |

## QLoRA Performance with Different Bits

| Number of Bits | Log of Perplexity (Fine-tuned LLAMA-2-13b on WikiText-2) |
|---|---|
| 16 | 1.63 |
| 8 | 1.64 |
| 4 | 1.63 |
| 3 | 1.63 |
| 2.5 | 2.82 |
| 2.25 | 6.40 |
| 2 | 6.37 |

## Results with 4-bit LoftQ of BART-large on XSum and CNN/DailyMail

| Quantization | Rank | Method | XSum | CNN/DailyMail |
|---|---|---|---|---|
| Full Precision | - | Lead-3 | 16.30/1.60/11.95 | 40.42/17.62/36.67 |
| - | - | Full FT | 45.14/22.27/37.25 | 44.16/21.28/40.90 |
| - | 8 | LoRA | 43.40/20.20/35.20 | 44.72/21.58/41.84 |
| - | 16 | LoRA | 43.95/20.72/35.68 | 45.03/21.84/42.15 |
| NF4 | 8 | QLoRA | 42.91/19.72/34.82 | 43.10/20.22/40.06 |
|  | 8 | LoftQ | 44.08/20.72/35.89 | 43.81/20.95/40.84 |
|  | 16 | QLoRA | 43.29/20.05/35.15 | 43.42/20.62/40.44 |
|  | 16 | LoftQ | 44.51/21.14/36.18 | 43.96/21.06/40.96 |
| Uniform | 8 | QLoRA | 41.84/18.71/33.74 | N.A. |
|  | 8 | LoftQ | 43.86/20.51/35.69 | 43.73/20.91/40.77 |
|  | 16 | QLoRA | 42.45/19.36/34.38 | 43.00/20.19/40.02 |
|  | 16 | LoftQ | 44.29/20.90/36.00 | 43.87/20.99/40.92 |

## Results with 2-bit LoftQ of BART-large on XSum and CNN/DailyMail using NF2 quantization

| Rank | Method | XSum | CNN/DailyMail |
|---|---|---|---|
| 8 | QLoRA | N.A. | N.A. |
|  | LoftQ | 39.63/16.65/31.62 | 42.24/19.44/29.04 |
| 16 | QLoRA | N.A. | N.A. |
|  | LoftQ | 40.81/17.85/32.80 | 42.52/19.81/39.51 |

