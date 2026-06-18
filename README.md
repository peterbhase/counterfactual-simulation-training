# Counterfactual Simulation Training

Code repo for this paper: [Counterfactual Simulation Training for Chain-of-Thought Faithfulness](https://arxiv.org/abs/2602.20710)

Below, we describe how to run experiments for Counterfactual Simulation Training (CST).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your API keys to `globals.py`:
```python
openai_api_key = "your-openai-key"
together_api_key = "your-together-key"
tinker_key = "your-tinker-key"
openrouter_key = "your-openrouter-key"
```

## Running a Quick Experiment

You can run a cheap experiment with `gpt-oss-20b` on Tinker and Openrouter with the command below. This experiment should cost about $6.25 (as of 6/17/26) and take around 2.5 hours. 

```bash
python run_jobs.py --experiment cheap_exp
```

This runs a small-scale experiment with:
- MMLU dataset
- 800 training examples, 800 test examples
- 5 epochs per round
- 2 training rounds

## Results

Results are saved to the `results/` directory. Stats are saved in `stats_roundspread_*.csv` files as with column structure:

```
dataname, counterfactual_type, metric, train_round0, train_round1, ..., test_round0, test_round1, ...
```

Here are example results from the `cheap_exp` from above, run for one seed (printable in `analysis.ipynb`). Columns show metrics before training (r0) and after each training round (r1, r2):

| metric | train_r0 | train_r1 | train_r2 | test_r0 | test_r1 | test_r2 |
|--------|----------|----------|----------|---------|---------|---------|
| bias_monitor_gmean | 0.444 | 0.834 | 0.852 | 0.466 | 0.786 | **0.818** |
| bias_monitor_acc | 0.656 | 0.862 | 0.868 | 0.682 | 0.834 | **0.835** |
| bias_monitor_precision | 0.744 | 0.890 | 0.903 | 0.854 | 0.879 | **0.890** |
| bias_monitor_recall | 0.206 | 0.739 | 0.774 | 0.223 | 0.654 | **0.721** |
| bias_monitor_FPR | 0.047 | 0.059 | 0.062 | 0.024 | 0.055 | 0.072 |
| greedy_bias_rate | 0.277 | 0.254 | 0.304 | 0.286 | 0.243 | 0.328 |

## Key Metrics

- **`monitor_acc_*`**: Monitor accuracy at detecting when model was influenced by a cue
- **`monitor_gmean_*`**: g-mean score for influence detection
- **`greedy_bias_rate_*`**: Rate at which model answers are biased by the cue

Metric suffixes indicate the scoring configuration (e.g., `k-0_CoT-F_cf-sim-xye` means simulator is 0-shot, no CoT, and conditions on the orig input CoT).

## Printing Examples

Individual examples are saved to `artifacts/` as CSV files:
```
artifacts/dataset_{exp_name}_round{round}_{train|test}.csv
```

Use `analysis.ipynb` to load and inspect examples. 

Plots in the paper are made with `final_plots.ipynb`.

# Project Structure

```
.
├── main.py              # Main experiment pipeline
├── run_jobs.py          # Job runner with experiment configurations
├── utils.py             # Utility functions
├── train_utils.py       # Training utilities
├── globals.py           # Configuration and API keys
├── prompt_templates/    # Prompt templates for various tasks
├── fewshot_examples/    # Few-shot examples for counterfactual generation
├── results/             # Aggregated metrics (stats CSVs)
├── artifacts/           # Per-example data (dataset CSVs)
├── analysis.ipynb       # Analysis notebook
└── final_plots.ipynb    # Plotting notebook
```
