# Prompt Templates

This folder contains documentation of all prompt structures used in the faithfulness-training codebase.

## Overview

| File | Function | Purpose |
|------|----------|---------|
| [01_fewshot_messages.md](01_fewshot_messages.md) | `build_fewshot_messages()` | MC question answering with optional CoT |
| [02_counterfactual_generation_messages.md](02_counterfactual_generation_messages.md) | `build_counterfactual_generation_messages()` | Generate counterfactual questions (6-shot) |
| [03_simulator_messages.md](03_simulator_messages.md) | `build_simulator_messages()` | Predict LLM behavior on counterfactuals |
| [04_monitor_messages.md](04_monitor_messages.md) | `build_monitor_messages()` | Detect bias cue influence (Yes/No) |
| [05_rewrite_messages.md](05_rewrite_messages.md) | `build_rewrite_messages()` | Rewrite reasoning for faithfulness |
| [06_verbalization_rewrite_messages.md](06_verbalization_rewrite_messages.md) | `build_verbalization_rewrite_messages()` | VFT-style bias verbalization |

## Model-Specific Thinking Tags

| Model Family | Opener | Closer |
|--------------|--------|--------|
| Default | `<think>` | `</think>` |
| DeepSeek V3.x, Gemini, GPT-5, Qwen3 | `<thinking>` | `</thinking>` |
| gpt-oss-20b/120b | `analysis` | `assistantfinal` |

## Bias Cue Types

See `globals.py` → `bias_description_dict` for full descriptions. Categories include:

- **Authority cues:** `cue_professor`, `cue_professor_2`, `sycophancy_authority`
- **Answer hints:** `answer_comment`, `answer_mark`, `answer_key`, `answer_key_user_help`
- **Social pressure:** `sycophancy_generic`, `sycophancy_school`, `llm_peer_pressure`
- **Emotional pressure:** `user_under_pressure`, `shut_down`
- **Backfire cues:** `backfire_*` variants (unreliable sources that should be ignored)
