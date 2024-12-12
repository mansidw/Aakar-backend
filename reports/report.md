LOFTQ (LoRA-Fine-Tuning-aware Quantization) is a novel quantization framework designed for large language models (LLMs) that require both quantization and LoRA fine-tuning. It addresses the performance gap observed when quantization and LoRA fine-tuning are applied together on pre-trained models. The framework integrates low-rank approximation with quantization to better align with the original high-precision pre-trained weights, providing a more effective initialization for LoRA fine-tuning. This approach significantly improves generalization in downstream tasks such as natural language understanding, question answering, summarization, and natural language generation.

LOFTQ outperforms existing quantization methods, particularly in low-bit scenarios like 2-bit and 2/4-bit mixed precision regimes. It achieves notable improvements in performance metrics, such as an 8% gain on MNLI and over 10% on SQuADv1.1 with 2-bit quantization methods. The framework is effective across different quantization methods and consistently surpasses the performance of QLoRA, especially in challenging low-bit environments.



## QLoRA Performance with Different Bits

| Number of Bits | Log of Perplexity (a) | Log of Perplexity (b) |
|---|---|---|
| 16 | 2.44 | 1.0 |
| 8 | 2.01 | 1.0 |
| 4 | 2.23 | 1.0 |
| 3 | 2.83 | 1.0 |
| 2.5 | 11.37 | 2.82 |
| 2.25 | 11.48 | 6.40 |
| 2 | 11.30 | 7.10 |

