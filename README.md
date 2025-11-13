# Enhancing Multi-Agent Debate Systems Performance via Confidence Expression

## Overview

This is the GitHub repo for **“Enhancing Multi-Agent Debate Systems Performance via Confidence Expression.”** 
We propose using **calibrated confidence expression** to improve the overall performance of Multi-Agent Debate (MAD) systems. 
Below is the workflow of our proposed MAD system with confidence expression, **ConfMAD**.

![Head-1](https://raw.githubusercontent.com/Enqurance/Figures/main/202511131530905.svg)

---

## Quick Start

Create a conda environment and install dependencies:

```bash
conda create -n debate python=3.12 -y
conda activate debate
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.7.2.post1
pip install -r requirements.txt
```

Export your API keys (ConfMAD supports OpenRouter for open-source models like LLaMA, Phi, etc.):

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export OPENROUTER_API_KEY=<YOUR_API_KEY>
export ANTHROPIC_API_KEY=<YOUR_API_KEY>
```

---

## Usage

### Basic Debate

Run a basic multi-agent debate **without** confidence scoring:

```bash
python llm_debate.py --task MATH --debate_turns 3 --debate_agents 2
```

### Supported Tasks

`BIGGSM`, `MATH`, `MMLU`, `BBH`

---

## Debate with Confidence Expression

### Self-Elicit Confidence

Model explicitly generates its own confidence score:

```bash
python llm_debate.py --debate_conf --conf_mode self_elicit --task MATH
```

### Length-Normalized Confidence

Confidence from normalized token probabilities:

```bash
python llm_debate.py --debate_conf --conf_mode length_norm --task MMLU
```

### Semantic Entropy Confidence (Not used in the paper since it is costly)

Confidence via semantic clustering of responses:

```bash
python llm_debate.py --debate_conf --conf_mode semantic_entropy --task BBH
```

### Sequence Probability Confidence

Confidence from raw sequence probabilities:

```bash
python llm_debate.py --debate_conf --conf_mode seq_prob --task MATH
```

---

## Confidence Calibration

### Train a Calibration Model

**Before debating with calibrated confidence, you must train a calibration model.**

```bash
python llm_debate.py --debate_conf --conf_mode self_elicit   --calibration_train --calibration_scheme platt   --calibration_conf self_elicit --task MATH --debate_turns 1
```

### Use a Pre-trained Calibration Model

```bash
python llm_debate.py --debate_conf --conf_mode self_elicit   --calibration --calibration_scheme platt   --calibration_conf self_elicit --task MATH
```

**Calibration schemes:** `platt`, `temperature`, `histogram`, `isotonic`

---

## Other Debate Modes

Chain-of-Thought Mode

```bash
python llm_debate.py --task MATH --cot
```

Multi-Persona Debate, [Liang et al.](https://arxiv.org/abs/2305.19118)

```bash
python llm_debate.py --task MATH --multi_persona --debate_agents 3
```

ChatEval, [Chan et al.](https://arxiv.org/abs/2308.07201)

```bash
python llm_debate.py --task BBH --chateval --debate_agents 3
```

Intervention, [Andrew Estornell and Yang Liu](https://proceedings.neurips.cc/paper_files/paper/2024/hash/32e07a110c6c6acf1afbf2bf82b614ad-Abstract-Conference.html)

```bash
python llm_debate.py --task MATH --intervention
```

## Advanced Options

### Categorical Confidence Scoring

```bash
python llm_debate.py --debate_conf --categorical   --categorical_bins 10 --conf_mode self_elicit --task MMLU
```

### Parallel Processing

```bash
python llm_debate.py --task MMLU --num_workers 10   --low_index 0 --up_index 2000
```

### Custom Output Directory

```bash
python llm_debate.py --task MATH --output_dir ./results
```

---

## Command-Line Arguments

### Core

- `--task` (str, default **BBH**): dataset
- `--debate_turns` (int, default **3**)
- `--debate_agents` (int, default **2**)
- `--debate_mode` (str, default **onebyone**): `onebyone` | `simultaneous`
- `--debate_conf` (flag): enable confidence-based debate

### Confidence

- `--conf_mode` (str, default **length_norm**): `self_elicit` | `length_norm` | `seq_prob` | `semantic_entropy` | `cluster`
- `--conf_type` (str, default **score**): `score` | `level`
- `--cluster_sample_times` (int, default **10**)

### Calibration

- `--calibration` (flag)
- `--calibration_train` (flag)
- `--calibration_scheme` (str, default **temperature**): `temperature` | `platt` | `histogram` | `isotonic`
- `--calibration_conf` (str)
- `--calibration_task` (str)
- `--calibration_overwrite` (flag)
- `--categorical` (flag)
- `--categorical_bins` (int, default **10**)
- `--top_logprobs` (int, default **10**)
- `--low_index_calibration` (int, default **0**)
- `--up_index_calibration` (int, default **1000**)

### Special Modes

- `--single`, `--cot`, `--intervention`, `--multi_persona`, `--chateval`, `--simultaneous` (all flags)

### Data / I/O

- `--low_index` (int, default **0**)
- `--up_index` (int, default **2000**)
- `--save_interval` (int, default **10**)
- `--num_workers` (int, default **5**)
- `--output_dir` (str, default **./result**)
- `--attempt_times` (int, default **2**)

---

## Output Format

Results are saved to:

```
{output_dir}/{TASK}/{DATE}/{TIME}.json
```

and include:

- question & ground-truth
- debate history for all turns
- confidence scores (if enabled)
- model responses
- metadata (arguments, timestamps, etc.)

---

## Examples

### Experiment 1: Basic Debate on MATH

```bash
python llm_debate.py --task MATH --debate_turns 3 --debate_agents 2
```

### Experiment 2: Confidence-Based Debate (Self-Elicit)

```bash
python llm_debate.py --debate_conf --conf_mode self_elicit   --task MMLU --debate_turns 3 --debate_agents 2
```

### Experiment 3: Train + Use Calibration

```bash
# Train
python llm_debate.py --debate_conf --conf_mode self_elicit   --calibration_train --calibration_scheme platt   --calibration_conf self_elicit --task MATH --debate_turns 1

# Inference with calibration
python llm_debate.py --debate_conf --conf_mode self_elicit   --calibration --calibration_scheme platt   --calibration_conf self_elicit --task MATH --debate_turns 3
```

### Experiment 4: Large-Scale Parallel

```bash
python llm_debate.py --task MMLU --debate_conf --conf_mode self_elicit   --num_workers 10 --up_index 5000 --save_interval 50
```

---

## Notes

- Train a calibration model first (`--calibration_train`) before using `--calibration`.
- `--calibration` and `--calibration_train` **cannot** be used together.
- `flash-attn` is optional; the system works without it.
- API rate limits may require tuning `--num_workers` and `--save_interval`.

---

## Citation

```bibtex
@article{lin2025enhancing,
  title   = {Enhancing Multi-Agent Debate System Performance via Confidence Expression},
  author  = {Lin, Zijie and Hooi, Bryan},
  journal = {arXiv preprint arXiv:2509.14034},
  year    = {2025}
}
```

