# TrajBias: Structural Biases in LLM-as-Judge Evaluation of Agent Trajectories

This repository contains the code and data for the paper:

> **TrajBias: Structural Biases in LLM-as-Judge Evaluation of Agent Trajectories**
> Yufeng He, The University of Hong Kong

## Key Findings

Through controlled perturbation experiments with 7 judge models from 7 families on coding agent trajectories, we discover:

1. **Error Recovery Penalty** (strongest): Trajectories with error recovery are systematically penalized (Cliff's delta up to 0.85, all p<0.001)
2. **Recency Bias**: Later errors receive harsher penalties than identical errors earlier
3. **Bidirectional Outcome Bias**: Outcome information contaminates process judgments in both directions, but opposing effects cancel in aggregate -- a methodological pitfall
4. **Absent Length Bias**: Trajectory length does not influence scores, contradicting text-domain verbosity bias
5. **Multidimensional Bias Robustness**: No single judge is universally least biased

## Dataset

Available on HuggingFace: [yufenghe/trajbias-benchmark](https://huggingface.co/datasets/yufenghe/trajbias-benchmark)

Contains:
- 2,400 controlled perturbation probe pairs (5 bias types)
- 20,000+ evaluation scores from 7 judge models
- Complete statistical analysis results
- Croissant metadata

## Repository Structure

```
scripts/
  bias_probes.py              # Generate perturbation probe pairs
  run_w3_full_eval.py         # Run LLM judge evaluations
  statistical_analysis_v2.py  # Statistical analysis (Cliff's delta, BH FDR)
  generate_figures.py         # Generate paper figures
  debiasing_experiments.py    # Debiasing strategy experiments
  prepare_hf_dataset.py       # Prepare HuggingFace dataset
```

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Judge Models Tested

| Judge | Family |
|-------|--------|
| Claude Sonnet 4.5 | Anthropic |
| GPT-5.2 | OpenAI |
| Gemini 3 Pro | Google |
| DeepSeek V3.2 | DeepSeek |
| Qwen3-235B | Alibaba |
| GLM-5 | Zhipu AI |
| Kimi K2.5 | Moonshot AI |

## Citation

```bibtex
@inproceedings{he2026trajbias,
  title={TrajBias: Structural Biases in LLM-as-Judge Evaluation of Agent Trajectories},
  author={He, Yufeng},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2026}
}
```

## License

MIT
