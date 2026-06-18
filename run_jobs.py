# run_jobs.py
"""
Simple job runner that loops through experiments and launches them.
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from itertools import product
import utils

# ===================== User-configurable =====================

# Base configuration (defaults are in main.py)
BASE = dict()

# Model-specific training configurations
MODEL_TO_BASE_TRAIN_CONFIG = {
}

# ===================== Experiment Definitions =====================

def get_experiment_commands(args) -> list[dict]:
    """
    Define experiment parameter sweeps. Returns list of config dicts.
    """
    base = dict(BASE)

    if args.experiment == "debug":
        epochs = 1
        train_rounds = 2
        simulator = "deepseek-v4-flash"
        score_config = "k-0_CoT-T_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye"
        n_sim_samples = 1
        axes = []
        # train_cues = "gpt_oss_bias_cues_train"
        train_cues = "cue_professor_2"
        # H2 reasoning-mixin smoke test config: small pool + small mixin so the
        # debug run actually exercises the mixin path (eval-only stemez +
        # distilled-trace mixing) end-to-end, without paying for a real run.
        reasoning_mixin_dataname = "mmlu-pro-stemez"
        # Switched from deepseek-v4-pro (eats tokens in a hidden reasoning
        # channel even at "low" effort) to Qwen3.5-397B-A17B with
        # enable_thinking=False (handled in utils.query_model_async).
        reasoning_mixin_model = "deepseek-v4-flash"
        # reasoning_mixin_model = "google/gemini-3.5-flash"
        reasoning_mixin_max_tokens = 8192
        reasoning_mixin_pool_size = 20
        reasoning_mixin_n = 10
        for dataset in [
                # "snli",
                # "ethics-commonsense",
                # "ethics-justice",
                # "mmlu-law",
                "mmlu"
            ]:
            n_train = 20
            n_test = 20
            for model, lr in [
                ("openai/gpt-oss-20b", 5e-4),
                # ("meta-llama/Llama-3.1-8B-Instruct", 5e-4),
                # ("meta-llama/Llama-3.1-70B-Instruct", 1e-4),
                # ("openai/gpt-oss-120b", 2e-4),
                # ("Qwen/Qwen3-235B-A22B-Instruct-2507", 1e-4),
            ]:
                for seed in [0]:
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    cache_name = "debug"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            # test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            test_only_datanames=[reasoning_mixin_dataname],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["PFT"],
                            case_based_rewards=[False],
                            cf_mix_perc=[0.1],
                            cf_add_perc=[-1],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                            rewriter_is_task_model=[False],
                            max_rewrite_rounds=[1],
                            reasoning_mixin_dataname=[reasoning_mixin_dataname],
                            reasoning_mixin_model=[reasoning_mixin_model],
                            reasoning_mixin_n=[reasoning_mixin_n],
                            reasoning_mixin_pool_size=[reasoning_mixin_pool_size],
                            reasoning_mixin_max_tokens=[reasoning_mixin_max_tokens],
                        )
                    )
    if args.experiment == "cheap_exp":
        epochs = 5
        train_rounds = 2
        # simulator = "qwen/qwen3-235b-a22b-2507"
        simulator = "deepseek-v4-flash"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 1
        # simulator = "deepseek/deepseek-chat-v3-0324"
        # score_config = "k-0_CoT-T_cf-sim-xye"
        # simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy"
        # n_sim_samples = 1
        axes = []
        for dataset in [
                "mmlu"
            ]:
            n_train = 800
            n_test = 800
            for model in [
                "openai/gpt-oss-20b",
            ]:
                lr = 5e-4
                for seed in [0]:
                    train_cues = "gpt_oss_bias_cues_train"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            cf_add_perc=[0.2],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                        )
                    )
    if args.experiment == "cue_direction":
        # Compare evidence-removal (orig=cued, cf=clean) vs evidence-addition (orig=clean, cf=cued).
        # Sweeps --cue_orig_has_cue across {True, False} for 5 seeds.
        epochs = 10
        train_rounds = 4
        simulator = "deepseek-v4-flash"
        # score_config = "k-0_CoT-F_cf-sim-xye"
        # simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        # n_sim_samples = 3
        score_config = "k-0_CoT-T_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy"
        n_sim_samples = 1
        axes = []
        for dataset in [
                "mmlu"
            ]:
            n_train = 1000
            n_test = 1000
            for model in [
                "openai/gpt-oss-120b",
            ]:
                # lr = 5e-4
                lr = 2e-4
                # for seed in [0, 1, 2, 3, 4]:
                for seed in [0]:
                    # for cue_orig_has_cue in [True]:
                    for cue_orig_has_cue in [True, False]:
                        # train_cues = "gpt_oss_bias_cues_train"
                        train_cues = "cue_professor_2"
                        # test_only_counterfactual_types = "gpt_oss_bias_cues_test"
                        cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                        axes.append(
                            dict(
                                exp_name_prefix=[f"cues_on_orig_{cue_orig_has_cue}"],
                                seed=[seed],
                                cache_mode=[cache_name],
                                train_counterfactual_types=[train_cues],
                                # test_only_counterfactual_types=[test_only_counterfactual_types],
                                train_datanames=[dataset],
                                task_model=[model],
                                n_cot_samples=[0],
                                simulator_model=[simulator],
                                n_train=[n_train],
                                n_test=[n_test],
                                epochs=[epochs],
                                train_rounds=[train_rounds],
                                score_based_on_config=[score_config],
                                simulator_sweep_configs=[simulator_sweep_configs],
                                n_sim_samples=[n_sim_samples],
                                rewrite_only_biased_data=[cue_orig_has_cue],
                                rewrite_only_backfire_data=[not cue_orig_has_cue],
                                loss_type=["unlikelihood"],
                                case_based_rewards=[True],
                                reward_multiplier=[5.0],
                                # cf_add_perc=[0.2],
                                cf_add_perc=[1.0],
                                lr=[lr],
                                max_data_to_load=[n_train + n_test],
                                cue_orig_has_cue=[cue_orig_has_cue],
                                # rewriter_is_task_model=[False],
                            )
                        )
    if args.experiment == "rewriter":
        epochs = 5
        n_train = 1000
        n_test = 2000
        train_rounds = 5
        simulator = "qwen/qwen3-235b-a22b-2507"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        train_cues = "gpt_oss_bias_cues_train"
        axes = []
        for dataset in ["mmlu"]:
            for model in ["openai/gpt-oss-120b"]:
                for simulator in [
                    "qwen/qwen3-235b-a22b-2507",
                    "deepseek/deepseek-chat-v3-0324",
                ]:
                    for rewriter_is_task_model in [False]:
                        for seed in [0, 1, 2, 3, 4]:
                            # sim model rewriter
                            axes.append(
                                dict(
                                    exp_name_prefix=[f"{args.experiment}_ritm-{rewriter_is_task_model}"],
                                    seed=[seed],
                                    cache_mode=[f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"],
                                    train_counterfactual_types=[train_cues],
                                    test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                                    train_datanames=[dataset],
                                    task_model=[model],
                                    n_cot_samples=[0],
                                    simulator_model=[simulator],
                                    n_train=[n_train],
                                    n_test=[n_test],
                                    epochs=[epochs],
                                    train_rounds=[train_rounds],
                                    score_based_on_config=[score_config],
                                    simulator_sweep_configs=[simulator_sweep_configs],
                                    n_sim_samples=[n_sim_samples],
                                    rewrite_only_biased_data=[True],
                                    loss_type=["unlikelihood"],
                                    case_based_rewards=[True],
                                    reward_multiplier=[5.0],
                                    cf_add_perc=[0.8],
                                    lr=[2e-4],
                                    max_data_to_load=[n_train + n_test],
                                    rewriter_is_task_model=[rewriter_is_task_model],
                                )
                            )
    if args.experiment == "ex_prompt":
        n_train = 1600
        n_test = 3200
        epochs = 0
        train_rounds = 0
        # simulator = "google/gemini-3-flash-preview"
        # simulator = "deepseek/deepseek-chat-v3-0324"
        # simulator = "qwen/qwen3-235b-a22b-2507"
        simulator = "deepseek/deepseek-chat-v3.1"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye"
        n_sim_samples = 1
        # score_config = "k-0_CoT-T_cf-sim-xye"
        # simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye"
        # n_sim_samples = 1
        # train_cues = "ex_cues"
        # train_cues = "ex_sycophancy"
        train_cues = "ex_cues_plus"
        axes = []
        for dataset in [
                "ethics-commonsense",
                # "ethics-AITA",
            ]:
            for model, lr in [
                ("gpt-5-mini-2025-08-07", -1),
                # ("openai/gpt-oss-120b", 2e-4),
                # ("Qwen/Qwen3-235B-A22B-Instruct-2507", 1e-4),
            ]:
                task_model_api = "openai"
                for seed in [0]:
                # for seed in [1]:
                    # main condition
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"],
                            train_counterfactual_types=[train_cues],
                            test_only_counterfactual_types=["ex_sycophancy_copy"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            cf_add_perc=[0.8],
                            lr=[lr],
                            rewrite_only_biased_data=[True],
                            reuse_data_across_cf_types=[False],
                            max_data_to_load=[4800],
                            use_tinker=[False],
                            use_api=[True],
                            task_model_api=[task_model_api],
                            rewriting_cots=[False],
                        )
                    )
    if args.experiment == "ex_run":
        n_train = 1600
        n_test = 3200
        epochs = 5
        train_rounds = 5
        # simulator = "google/gemini-3-flash-preview"
        simulator = "deepseek/deepseek-chat-v3-0324"
        # simulator = "qwen/qwen3-235b-a22b-2507"
        api_name = "openrouter"
        # simulator = "gpt-4.1-mini-2025-04-14"
        # api_name = "openai"
        # score_config = "k-0_CoT-T_cf-sim-xye"
        # simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy"
        # n_sim_samples = 1
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        train_cues = "ex_cues_plus"
        # train_cues = "ex_cues"
        # train_cues = "ex_sycophancy"
        axes = []
        for dataset in [
                # "ethics-commonsense",
                "ethics-AITA",
            ]:
            for model, lr in [
                ("openai/gpt-oss-120b", 2e-4),
                # ("Qwen/Qwen3-235B-A22B-Instruct-2507", 1e-4),
            ]:
                for seed in [0]:
                # for seed in [1]:
                    # main condition
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"],
                            train_counterfactual_types=[train_cues],
                            test_only_counterfactual_types=["ex_sycophancy_copy"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            # loss_type=["PFT"],
                            cf_add_perc=[0.8],
                            lr=[lr],
                            rewrite_only_biased_data=[True],
                            reuse_data_across_cf_types=[False],
                            max_data_to_load=[4800],
                            api_name=[api_name],
                        )
                    )
    if args.experiment == "lead_fig":
        train_rounds = 6
        n_sim_samples = 3
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        axes = []
        for dataset in ["snli"]:
        # for dataset in ["mmlu"]:
            for model in ["openai/gpt-oss-120b"]:
                for seed in [0, 1, 2, 3, 4]:
                    # 1. cue-based cfs
                    n_train = 1000
                    n_test = 2000
                    train_cues = "gpt_oss_bias_cues_train"
                    simulator = "deepseek/deepseek-chat-v3-0324"
                    # simulator = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
                    # api_name = "together"
                    api_name = "openrouter"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            api_name=[api_name],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[5],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[2.0],
                            cf_add_perc=[0.8],
                            lr=[2e-4],
                            max_data_to_load=[3000],
                            force_rerun=[seed==0]
                        )
                    )
                    # # 2. model-generated cfs
                    # for n_train in [1000, 3000]:
                    # # n_train = 3000
                    #     n_test = 1000
                    #     train_cues = "model_based"
                    #     simulator = "qwen/qwen3-235b-a22b-2507"
                    #     cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    #     api_name = "openrouter"
                    #     axes.append(
                    #         dict(
                    #             exp_name_prefix=[f"{args.experiment}"],
                    #             seed=[seed],
                    #             cache_mode=[cache_name],
                    #             train_counterfactual_types=[train_cues],
                    #             train_datanames=[dataset],
                    #             task_model=[model],
                    #             n_cot_samples=[0],
                    #             simulator_model=[simulator],
                    #             api_name=[api_name],
                    #             n_train=[n_train],
                    #             n_test=[n_test],
                    #             epochs=[epochs],
                    #             train_rounds=[train_rounds],
                    #             score_based_on_config=[score_config],
                    #             simulator_sweep_configs=[simulator_sweep_configs],
                    #             n_sim_samples=[n_sim_samples],
                    #             loss_type=["PFT"],
                    #             positives_are_actively_helpful=[True],
                    #             cf_add_perc=[0.8],
                    #             lr=[2e-4],
                    #             explanation_specific_counterfactuals=[False],
                    #             cf_rejection_sampling=[True],
                    #         )
                    #     )
    if args.experiment == "prompting_qwen":
        epochs = 0
        train_rounds = 0
        simulator = "deepseek/deepseek-chat-v3-0324"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        api_name = "openrouter"
        n_sim_samples = 3
        n_train = 0
        n_test = 2000
        axes = []
        for dataset in [
            # "snli",
            "mmlu",
        ]:
            for model in ["qwen/qwen3-235b-a22b-2507"]:
                for seed in [0, 1, 2, 3, 4]:
                    # 1. cue-based cfs
                    for strategy in ["default", "exhaustive", "principles", "faithful_def", "test_description"]:
                        if dataset == "mmlu":
                            train_cues = "gpt_oss_bias_cues_train"
                            cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                            axes.append(
                                dict(
                                    exp_name_prefix=[f"{args.experiment}_{strategy}"],
                                    seed=[seed],
                                    cache_mode=[cache_name],
                                    train_counterfactual_types=[train_cues],
                                    test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                                    train_datanames=[dataset],
                                    task_model=[model],
                                    n_cot_samples=[0],
                                    simulator_model=[simulator],
                                    n_train=[n_train],
                                    n_test=[n_test],
                                    epochs=[epochs],
                                    train_rounds=[train_rounds],
                                    score_based_on_config=[score_config],
                                    simulator_sweep_configs=[simulator_sweep_configs],
                                    n_sim_samples=[n_sim_samples],
                                    max_data_to_load=[3000],
                                    reasoning_instructions=[strategy],
                                    use_tinker=[False],
                                    use_api=[True],
                                    task_model_api=["openrouter"],
                                )
                            )
                        # 2. model-generated cfs
                        if dataset == "snli":
                            train_cues = "model_based"
                            cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                            axes.append(
                                dict(
                                    exp_name_prefix=[f"{args.experiment}_{strategy}"],
                                    seed=[seed],
                                    cache_mode=[cache_name],
                                    train_counterfactual_types=[train_cues],
                                    train_datanames=[dataset],
                                    task_model=[model],
                                    n_cot_samples=[0],
                                    n_cf_samples=[3],
                                    simulator_model=[simulator],
                                    api_name=[api_name],
                                    n_train=[n_train],
                                    n_test=[n_test],
                                    epochs=[epochs],
                                    train_rounds=[train_rounds],
                                    score_based_on_config=[score_config],
                                    simulator_sweep_configs=[simulator_sweep_configs],
                                    n_sim_samples=[n_sim_samples],
                                    explanation_specific_counterfactuals=[False],
                                    cf_rejection_sampling=[True],
                                    max_data_to_load=[3000],
                                    reasoning_instructions=[strategy],
                                    use_tinker=[False],
                                    use_api=[True],
                                    task_model_api=["openrouter"],
                                )
                            )
    if args.experiment == "prompting_gpt-oss":
        epochs = 0
        train_rounds = 0
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        n_train = 0
        n_test = 2000
        axes = []
        for dataset in [
            "mmlu",
            # "snli",
        ]:
            for model in ["openai/gpt-oss-120b"]:
                # for seed in [0]:
                for seed in [0, 1, 2, 3, 4]:
                    # 1. cue-based cfs
                    for reasoning_effort in ["low", "medium", "high"]:
                        train_cues = "gpt_oss_bias_cues_train"
                        simulator = "deepseek/deepseek-chat-v3-0324"
                        # simulator = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
                        # simulator = "qwen/qwen3-235b-a22b-2507"
                        api_name = "together" if "Qwen" in simulator else "openrouter"
                        cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                        axes.append(
                            dict(
                                exp_name_prefix=[f"{args.experiment}_{reasoning_effort}"],
                                seed=[seed],
                                cache_mode=[cache_name],
                                train_counterfactual_types=["gpt_oss_bias_cues_train"],
                                test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                                train_datanames=[dataset],
                                task_model=[model],
                                n_cot_samples=[0],
                                simulator_model=[simulator],
                                n_train=[n_train],
                                n_test=[n_test],
                                epochs=[epochs],
                                train_rounds=[train_rounds],
                                score_based_on_config=[score_config],
                                simulator_sweep_configs=[simulator_sweep_configs],
                                n_sim_samples=[n_sim_samples],
                                task_model_api=["together"],
                                api_name=[api_name],
                                max_data_to_load=[3000],
                                reasoning_effort=[reasoning_effort],
                                use_tinker=[False],
                                use_api=[True],
                            )
                        )
                        # 2. model-generated cfs
                        train_cues = "model_based"
                        cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                        axes.append(
                            dict(
                                exp_name_prefix=[f"{args.experiment}_{reasoning_effort}"],
                                seed=[seed],
                                cache_mode=[cache_name],
                                train_counterfactual_types=[train_cues],
                                train_datanames=[dataset],
                                task_model=[model],
                                n_cot_samples=[0],
                                n_cf_samples=[3],
                                simulator_model=[simulator],
                                n_train=[n_train],
                                n_test=[n_test],
                                epochs=[epochs],
                                train_rounds=[train_rounds],
                                score_based_on_config=[score_config],
                                simulator_sweep_configs=[simulator_sweep_configs],
                                n_sim_samples=[n_sim_samples],
                                explanation_specific_counterfactuals=[False],
                                cf_rejection_sampling=[True],
                                task_model_api=["together"],
                                api_name=[api_name],
                                max_data_to_load=[3000],
                                reasoning_effort=[reasoning_effort],
                                use_tinker=[False],
                                use_api=[True],
                            )
                        )
    if args.experiment == "cue_pivot":
        epochs = 10
        train_rounds = 4
        simulator = "deepseek-v4-flash"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        # simulator = "deepseek/deepseek-chat-v3-0324"
        # score_config = "k-0_CoT-T_cf-sim-xye"
        # simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy"
        # n_sim_samples = 1
        axes = []
        for dataset in [
                "mmlu"
                # "chaosNLI",
                # "snli",
            ]:
            n_train = 1000
            n_test = 1000
            # n_test = 900
            for model in [
                "openai/gpt-oss-20b",
                # "openai/gpt-oss-120b",
            ]:
                lr = 5e-4
                # lr = 2e-4
                for seed in [0]:
                    train_cues = "gpt_oss_bias_cues_train"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            # test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            cf_add_perc=[0.8],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                        )
                    )
    if args.experiment == "cross_dataset":
        # H3 (rebuttal): train CST on MMLU cue-based CFs only, then
        # evaluate monitor G-mean / recall / FPR on held-out datasets
        # (SNLI, ETHICS commonsense, ETHICS justice, MedQA, ARC) without
        # any additional training. Cue-based only — cues are
        # dataset-agnostic, so this is the cleanest cross-dataset
        # transfer test. All non-dataset args mirror `cue_pivot` for
        # direct comparability.
        epochs = 10
        train_rounds = 4
        simulator = "deepseek-v4-flash"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        # score_config = "k-0_CoT-T_cf-sim-xye"
        # simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy"
        # n_sim_samples = 1
        axes = []
        for dataset in [
                "mmlu"
            ]:
            n_train = 1000
            n_test = 1000
            for model in [
                "openai/gpt-oss-20b",
            ]:
                lr = 2e-4
                for seed in [0]:
                    train_cues = "gpt_oss_bias_cues_train"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}_transfer"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            train_datanames=[dataset],
                            # held-out eval datasets (no training on these)
                            test_only_datanames=[
                                "snli ethics-commonsense medqa arc"
                            ],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            cf_add_perc=[0.8],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                        )
                    )
    if args.experiment == "reasoning_mixin":
        # H2 (rebuttal): test whether mixing distilled reasoning traces from a
        # strong reasoner into the CST positive-example pool preserves
        # reasoning capability on a held-out reasoning-heavy benchmark
        # (mmlu-pro-stemez, 2-way MC).
        #
        # Two arms:
        #   - control:   reasoning_mixin_n=0 (no mixin) → measures CST's
        #                raw effect on Stemez accuracy.
        #   - treatment: reasoning_mixin_n>0 with deepseek/deepseek-v4-pro
        #                traces on mmlu-pro-stemez → should recover Stemez
        #                accuracy while preserving monitor G-mean.
        #
        # All other args mirror `cue_pivot` so results are directly comparable.
        epochs = 10
        train_rounds = 4
        simulator = "deepseek-v4-flash"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        # score_config = "k-0_CoT-T_cf-sim-xye"
        # simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy"
        # n_sim_samples = 1
        # mixin config (treatment arm).
        # NOTE: stemez train / test split is now carved disjointly in
        # load_datasets_and_configs (load pool_size + n_test rows once, split
        # by id). pool_size is the raw stemez-row budget sent to the reasoning
        # model; correct-only filter shrinks it; final mixin = min(correct,
        # reasoning_mixin_n). Need ~20% headroom over n for filter loss.
        # filter_src_to_str='stemez' -- if pool+n_test exceeds available,
        # load_datasets_and_configs prints a warning and truncates.
        reasoning_mixin_dataname = "mmlu-pro-stemez"
        # reasoning_mixin_model = "deepseek/deepseek-v4-pro"
        reasoning_mixin_model = "qwen/qwen3.5-397b-a17b"
        # reasoning_mixin_model = "openai/gpt-oss-120b"
        reasoning_mixin_max_tokens = 8192
        # 1000 mixin target, 1200 pool gives headroom for ~17% filter miss.
        reasoning_mixin_pool_size = 1200
        reasoning_mixin_n_target = 1000
        axes = []
        for dataset in [
                "arc"
            ]:
            n_train = 1000
            # n_test applies to BOTH the mmlu eval split and the disjoint
            # stemez test split (carved out of pool_size + n_test).
            n_test = 1000
            for model in [
                "openai/gpt-oss-120b",
            ]:
                lr = 2e-4
                for seed in [0]:
                    train_cues = "gpt_oss_bias_cues_train"
                    cache_name = "mixin"
                    # Sweep two arms over reasoning_mixin_n.
                    # 0 = control (no mixin), >0 = treatment.
                    for mixin_n in [reasoning_mixin_n_target, 0]:
                        axes.append(
                            dict(
                                exp_name_prefix=[f"{args.experiment}_v2_n{mixin_n}"],
                                seed=[seed],
                                cache_mode=[cache_name],
                                train_counterfactual_types=[train_cues],
                                train_datanames=[dataset],
                                # add the reasoning benchmark as a held-out eval
                                test_only_datanames=[reasoning_mixin_dataname],
                                task_model=[model],
                                n_cot_samples=[0],
                                simulator_model=[simulator],
                                n_train=[n_train],
                                n_test=[n_test],
                                epochs=[epochs],
                                train_rounds=[train_rounds],
                                score_based_on_config=[score_config],
                                simulator_sweep_configs=[simulator_sweep_configs],
                                n_sim_samples=[n_sim_samples],
                                rewrite_only_biased_data=[True],
                                loss_type=["unlikelihood"],
                                case_based_rewards=[True],
                                reward_multiplier=[5.0],
                                cf_add_perc=[0.1],
                                lr=[lr],
                                # NOTE: needs to cover BOTH the main mmlu
                                # train/test load (n_train + n_test) and the
                                # disjoint stemez pool+test load
                                # (pool_size + n_test). The loader uses
                                # max(needed, max_data_to_load), so set this
                                # high enough that both fit.
                                max_data_to_load=[max(n_train + n_test, reasoning_mixin_pool_size + n_test)],
                                # mixin flags. n=0 disables the mixin entirely
                                # in main.py (no trace generation, no append).
                                reasoning_mixin_dataname=[reasoning_mixin_dataname],
                                reasoning_mixin_model=[reasoning_mixin_model],
                                reasoning_mixin_n=[mixin_n],
                                reasoning_mixin_pool_size=[reasoning_mixin_pool_size],
                                reasoning_mixin_max_tokens=[reasoning_mixin_max_tokens],
                            )
                        )
    if args.experiment == "debug_mixin":
        # Tiny smoke-test version of the `reasoning_mixin` (v2) experiment.
        # All train / test / pool sizes shrunk to 20 so the full pipeline
        # (CST training + disjoint stemez eval + distilled-trace mixin) runs
        # end-to-end in minutes. Mirrors `reasonidng_mixin` config otherwise.
        epochs = 1
        train_rounds = 2
        simulator = "deepseek-v4-flash"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 1
        # mixin config (treatment arm) -- tiny.
        reasoning_mixin_dataname = "mmlu-pro-stemez"
        reasoning_mixin_model = "qwen/qwen3.5-397b-a17b"
        reasoning_mixin_max_tokens = 8192
        reasoning_mixin_pool_size = 40
        reasoning_mixin_n_target = 20
        axes = []
        for dataset in [
                "arc"
            ]:
            # NOTE: keep n_train / n_test >= len(gpt_oss_bias_cues_train) = 6
            # with comfortable per-chunk size. utils.chunk_dataset uses
            # ceil(n / n_chunks) and produces an empty 6th chunk when n=20
            # (5 chunks of 4 fill it), which then crashes evaluate_faithfulness.
            # n=40 -> ceil(40/6)=7, chunks 6*7=42 capped to 40, last chunk size 5. Fine.
            n_train = 40
            n_test = 40
            for model in [
                "openai/gpt-oss-120b",
            ]:
                lr = 2e-4
                for seed in [0]:
                    train_cues = "gpt_oss_bias_cues_train"
                    cache_name = "mixin_debug"
                    # Sweep two arms: 0 = control (no mixin), >0 = treatment.
                    for mixin_n in [reasoning_mixin_n_target, 0]:
                        axes.append(
                            dict(
                                exp_name_prefix=[f"{args.experiment}_n{mixin_n}"],
                                seed=[seed],
                                cache_mode=[cache_name],
                                train_counterfactual_types=[train_cues],
                                train_datanames=[dataset],
                                test_only_datanames=[reasoning_mixin_dataname],
                                task_model=[model],
                                n_cot_samples=[0],
                                simulator_model=[simulator],
                                n_train=[n_train],
                                n_test=[n_test],
                                epochs=[epochs],
                                train_rounds=[train_rounds],
                                score_based_on_config=[score_config],
                                simulator_sweep_configs=[simulator_sweep_configs],
                                n_sim_samples=[n_sim_samples],
                                rewrite_only_biased_data=[True],
                                loss_type=["unlikelihood"],
                                case_based_rewards=[True],
                                reward_multiplier=[5.0],
                                cf_add_perc=[0.1],
                                lr=[lr],
                                max_data_to_load=[max(n_train + n_test, reasoning_mixin_pool_size + n_test)],
                                reasoning_mixin_dataname=[reasoning_mixin_dataname],
                                reasoning_mixin_model=[reasoning_mixin_model],
                                reasoning_mixin_n=[mixin_n],
                                reasoning_mixin_pool_size=[reasoning_mixin_pool_size],
                                reasoning_mixin_max_tokens=[reasoning_mixin_max_tokens],
                            )
                        )
    if args.experiment == "cot_prefill":
        epochs = 10
        train_rounds = 4
        simulator = "deepseek-v4-flash"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        # simulator = "deepseek/deepseek-chat-v3-0324"
        # score_config = "k-0_CoT-T_cf-sim-xye"
        # simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy"
        # n_sim_samples = 1
        axes = []
        for dataset in [
                "mmlu"
            ]:
            n_train = 1000
            n_test = 1000
            for model in [
                "openai/gpt-oss-20b",
            ]:
                lr = 2e-4
                for seed in [0]:
                    train_cues = "gpt_oss_bias_cues_train"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            # test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            cf_add_perc=[0.8],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                            run_prefill_eval=[True],
                        )
                    )
    if args.experiment == "main_sweep":
        train_rounds = 6
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        axes = []
        for dataset in [
                # "ethics-commonsense",
                # "ethics-justice",
                # "mmlu-pro-law",
                "snli",
                # "mmlu",
            ]:
            if "mmlu-pro-law" in dataset:
                n_train = 640
                n_test = 320
            else:
                n_train = 1000
                n_test = 2000
            for model in [
                # "openai/gpt-oss-120b",
                "Qwen/Qwen3-235B-A22B-Instruct-2507",
            ]:
                lr = 2e-4 if "gpt-oss" in model else 1e-4
                for seed in [0, 1, 2, 3, 4]:
                    # if "mmlu" in dataset and seed < 3:
                    #     continue
                    # if "commonsense" in dataset and seed < 2:
                    #     continue
                    # if "snli" in dataset and seed < 2:
                    #     continue
                    for simulator in [
                        # "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                        # "qwen/qwen3-235b-a22b-2507",
                        "deepseek/deepseek-chat-v3-0324",
                        # "deepseek/deepseek-chat-v3.1",
                    ]:
                        api_name = "together" if "Qwen" in simulator else "openrouter"
                        # 1. cue-based cfs
                        # train_cues = "gpt_oss_bias_cues_train"
                        # cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                        # axes.append(
                        #     dict(
                        #         exp_name_prefix=[f"{args.experiment}"],
                        #         seed=[seed],
                        #         cache_mode=[cache_name],
                        #         train_counterfactual_types=[train_cues],
                        #         test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                        #         train_datanames=[dataset],
                        #         task_model=[model],
                        #         n_cot_samples=[0],
                        #         simulator_model=[simulator],
                        #         n_train=[n_train],
                        #         n_test=[n_test],
                        #         epochs=[5],
                        #         train_rounds=[train_rounds],
                        #         score_based_on_config=[score_config],
                        #         simulator_sweep_configs=[simulator_sweep_configs],
                        #         n_sim_samples=[n_sim_samples],
                        #         rewrite_only_biased_data=[True],
                        #         loss_type=["unlikelihood"],
                        #         case_based_rewards=[True],
                        #         reward_multiplier=[5.0],
                        #         cf_add_perc=[0.8],
                        #         lr=[lr],
                        #         max_data_to_load=[n_train + n_test],
                        #         api_name=[api_name],
                        #     )
                        # )
                        # 2. model-generated cfs
                        train_cues = "model_based"
                        cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                        axes.append(
                            dict(
                                exp_name_prefix=[f"{args.experiment}"],
                                seed=[seed],
                                cache_mode=[cache_name],
                                train_counterfactual_types=[train_cues],
                                train_datanames=[dataset],
                                task_model=[model],
                                n_cot_samples=[15],
                                n_cf_samples=[3],
                                rewriting_cots=[True],
                                simulator_model=[simulator],
                                n_train=[n_train],
                                n_test=[n_test],
                                epochs=[20 if "gpt-oss" in model else 10],
                                train_rounds=[train_rounds],
                                score_based_on_config=[score_config],
                                simulator_sweep_configs=[simulator_sweep_configs],
                                n_sim_samples=[n_sim_samples],
                                loss_type=["unlikelihood"],
                                case_based_rewards=[True],
                                reward_multiplier=[5],
                                unlike_lambda=[0.4],
                                cf_add_perc=[0.2],
                                lr=[2e-4],
                                explanation_specific_counterfactuals=[False],
                                cf_rejection_sampling=[True],
                                max_data_to_load=[3000],
                                api_name=[api_name],
                            )
                        )
    if args.experiment == "cue_influence":
        epochs = 5
        train_rounds = 0
        score_config = "k-0_CoT-F_cf-sim-xy"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xy"
        train_cues = "sweep_list"
        n_sim_samples = 1
        cue_corrupt_rate = 0.5
        axes = []
        for dataset in [
                # "snli",
                "ethics-commonsense",
                # "ethics-justice",
                # "mmlu-law",
                # "mmlu"
            ]:
            n_train = 0
            n_test = 2000
            for model in [
                "openai/gpt-oss-120b",
                "qwen/qwen3-235b-a22b-2507",
            ]:
                for seed in [0]: 
                    # 1. cue-based cfs
                    simulator = "qwen/qwen3-235b-a22b-2507"
                    api_name = "openrouter"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}_cr{cue_corrupt_rate}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            main_temperature=[0.7],
                            main_top_p=[0.95],
                            n_cf_samples=[2],
                            simulator_model=[simulator],
                            api_name=[api_name],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            cue_corrupt_rate=[cue_corrupt_rate],
                            max_data_to_load=[n_train + n_test],
                            reuse_data_across_cf_types=[True],
                            use_tinker=[False],
                            use_api=[True],
                        )
                    )
    if args.experiment == "backfire_cues":
        epochs = 10
        train_rounds = 5
        # simulator = "qwen/qwen3-235b-a22b-2507"
        simulator = "deepseek/deepseek-chat-v3-0324"
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        axes = []
        for dataset in [
                "mmlu"
            ]:
            n_train = 1000
            n_test = 2000
            for model in [
                "openai/gpt-oss-120b",
                "Qwen/Qwen3-235B-A22B-Instruct-2507",
            ]:
                lr = 2e-4
                # for seed in [0]:
                for seed in [0, 1, 2, 3, 4]:
                    train_cues = "gpt_oss_backfire_cues"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[False],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            cf_add_perc=[0.8],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                        )
                    )
    if args.experiment == "RL":
        # epochs = 10
        epochs = 5
        train_rounds = 5
        # simulator = "qwen/qwen3-235b-a22b-2507"
        # simulator = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
        # score_config = "k-0_CoT-F_cf-sim-xye"
        # simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        # n_sim_samples = 3
        # api_name = "together"
        simulator = "deepseek/deepseek-chat-v3-0324"
        score_config = "k-0_CoT-T_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy"
        n_sim_samples = 1
        api_name = "openrouter"
        axes = []
        for dataset in [
                "mmlu"
            ]:
            n_train = 1000
            n_test = 2000
            for model in [
                "openai/gpt-oss-120b",
            ]:
                lr = 2e-4
                # for seed in [0]:
                # for seed in [0, 1, 2, 3, 4]:
                # for seed in [1]:
                for seed in [3, 4]:
                    train_cues = "gpt_oss_bias_cues_train"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[7],
                            rewriting_cots=[False],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            # loss_type=["PFT"],
                            # case_based_rewards=[False],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            # unlike_lambda=[0.1],
                            cf_add_perc=[0.8],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                            api_name=[api_name],
                        )
                    )
    if args.experiment == "scaling_model":
        train_rounds = 5
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        axes = []
        for dataset in [
                # "snli",
                # "ethics-commonsense",
                # "ethics-justice",
                # "mmlu-law",
                "mmlu"
            ]: 
            if "mmlu-law" in dataset:
                n_train = 1000
                n_test = 600
            else:
                n_train = 1000
                n_test = 2000
            for model, lr, epochs in [
                # ("Qwen/Qwen3-4B-Instruct-2507", 5e-4, 10),
                ("Qwen/Qwen3-30B-A3B-Instruct-2507", 5e-4, 5),
                # ("Qwen/Qwen3-235B-A22B-Instruct-2507", 1e-4, 5),
            ]:
                for seed in [0, 1, 2, 3, 4]:
                    if "30B" in model and not seed in [2,3]:
                        continue
                    train_cues = "gpt_oss_bias_cues_train"
                    simulator = "deepseek/deepseek-chat-v3-0324"
                    api_name = "openrouter"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            rewriting_cots=[True],
                            simulator_model=[simulator],
                            api_name=[api_name],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            cf_add_perc=[0.8],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                            rewriter_is_task_model=[False],
                        )
                    )
    if args.experiment == "scaling_data":
        train_rounds = 5
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        axes = []
        for dataset in [
                "mmlu"
            ]:
            n_test = 2000
            for model, n_train, epochs in [
                ("openai/gpt-oss-120b", 50, 11),
                ("openai/gpt-oss-120b", 125, 5),
                ("openai/gpt-oss-120b", 500, 5),
            ]:
                for seed in [0, 1, 2, 3, 4]:
                    if n_train == 250 and seed != 1:
                        continue
                    train_cues = "gpt_oss_bias_cues_train"
                    simulator = "qwen/qwen3-235b-a22b-2507"
                    api_name = "openrouter"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            rewriting_cots=[True],
                            simulator_model=[simulator],
                            api_name=[api_name],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            cf_add_perc=[0.8],
                            lr=[2e-4],
                            max_data_to_load=[3000],
                        )
                    )
    if args.experiment == "Rxye":
        epochs = 5
        train_rounds = 5
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        axes = []
        for dataset in [
                # "snli",
                # "ethics-commonsense",
                # "ethics-justice",
                # "mmlu-law",
                "mmlu"
            ]:
            if "mmlu-law" in dataset:
                n_train = 1000
                n_test = 600
            else:
                n_train = 1000
                n_test = 2000
            for model in [
                "openai/gpt-oss-120b",
                # "Qwen/Qwen3-235B-A22B-Instruct-2507",
            ]:
                lr = 2e-4 if "gpt-oss" in model else 1e-4
                for seed in [0, 1, 2, 3, 4]:
                    train_cues = "gpt_oss_bias_cues_train"
                    simulator = "qwen/qwen3-235b-a22b-2507"
                    api_name = "openrouter"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            rewriting_cots=[True],
                            simulator_model=[simulator],
                            api_name=[api_name],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["PFT"],
                            cf_add_perc=[0.8],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                        )
                    )
    if args.experiment == "offline":
        epochs = 20
        train_rounds = 1
        score_config = "k-0_CoT-F_cf-sim-xye"
        simulator_sweep_configs = "k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy"
        n_sim_samples = 3
        axes = []
        for dataset in [
                # "snli",
                # "ethics-commonsense",
                # "ethics-justice",
                # "mmlu-law",
                "mmlu"
            ]:
            if "mmlu-law" in dataset:
                n_train = 1000
                n_test = 600
            else:
                n_train = 1000
                n_test = 2000
            for model in [
                "openai/gpt-oss-120b",
                # "Qwen/Qwen3-235B-A22B-Instruct-2507",
            ]:
                lr = 2e-4 if "gpt-oss" in model else 1e-4
                for seed in [0, 1, 2, 3, 4]:
                    train_cues = "gpt_oss_bias_cues_train"
                    simulator = "qwen/qwen3-235b-a22b-2507"
                    api_name = "openrouter"
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            rewriting_cots=[True],
                            simulator_model=[simulator],
                            api_name=[api_name],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["unlikelihood"],
                            case_based_rewards=[True],
                            reward_multiplier=[5.0],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                        )
                    )
    if args.experiment == "VFT":
        epochs = 20
        train_rounds = 1
        # simulator = "qwen/qwen3-235b-a22b-2507"
        simulator = "deepseek/deepseek-chat-v3-0324"
        score_config = "k-0_CoT-T_yesno-xye"
        simulator_sweep_configs = "k-0_CoT-T_yesno-xye"
        n_sim_samples = 1
        axes = []
        train_cues = "gpt_oss_bias_cues_train"
        for dataset in [
                # "snli",
                # "ethics-commonsense",
                # "ethics-justice",
                # "mmlu-law",
                "mmlu"
            ]:
            if "mmlu-law" in dataset:
                n_train = 1000
                n_test = 600
            else:
                n_train = 1000
                n_test = 2000
            for model, lr in [
                # ("meta-llama/Llama-3.1-8B-Instruct", 5e-4),
                # ("meta-llama/Llama-3.1-70B-Instruct", 1e-4),
                ("openai/gpt-oss-120b", 2e-4),
                ("Qwen/Qwen3-235B-A22B-Instruct-2507", 1e-4),
            ]:
                for seed in [0, 1, 2, 3, 4]:
                    cache_name = f"{train_cues}_{dataset}_{utils.model_to_string(model)}_{utils.model_to_string(simulator)}"
                    axes.append(
                        dict(
                            exp_name_prefix=[f"{args.experiment}_v2"],
                            seed=[seed],
                            cache_mode=[cache_name],
                            train_counterfactual_types=[train_cues],
                            test_only_counterfactual_types=["gpt_oss_bias_cues_test"],
                            train_datanames=[dataset],
                            task_model=[model],
                            n_cot_samples=[0],
                            simulator_model=[simulator],
                            n_train=[n_train],
                            n_test=[n_test],
                            epochs=[epochs],
                            train_rounds=[train_rounds],
                            score_based_on_config=[score_config],
                            simulator_sweep_configs=[simulator_sweep_configs],
                            n_sim_samples=[n_sim_samples],
                            rewrite_only_biased_data=[True],
                            loss_type=["PFT"],
                            case_based_rewards=[False],
                            cf_add_perc=[0.1],
                            lr=[lr],
                            max_data_to_load=[n_train + n_test],
                            rewriter_is_task_model=[False],
                            max_rewrite_rounds=[1],
                            rewrite_only_FNs=[True],
                        )
                    )
                    


    jobs = []
    # Handle axes as either a dict or list of dicts
    axes_list = axes if isinstance(axes, list) else [axes]
    
    for axes_dict in axes_list:
        keys, values = zip(*axes_dict.items()) if axes_dict else ([], [])
        for combo in product(*values) if values else [()]:
            cfg = dict(base)
            for k, v in zip(keys, combo):
                cfg[k] = v
            jobs.append(cfg)
    
    return jobs

# ===================== Command Building =====================

def args_to_cmd(a: dict) -> str:
    """
    Build a shell command for main.py from a config dict.
    """
    parts = [sys.executable, "main.py"]
    
    # Add model-specific base config first
    model = a.get("task_model")
    base_train_cfg = MODEL_TO_BASE_TRAIN_CONFIG.get(model, {})
    for k, v in base_train_cfg.items():
        if v is None:
            continue
        parts.append(f"--{k} {v}")
    
    # Add experiment-specific args
    for k, v in a.items():
        if v is None:
            continue
        parts.append(f"--{k} {v}")
    
    return " ".join(parts)


def run_job(args, cfg: dict, job_num: int, total_jobs: int):
    """
    Launch a single experiment job inside the isolated workspace.
    """
    cmd = args_to_cmd(cfg)
    print(f"\n{'='*60}")
    print(f"Job {job_num}/{total_jobs}")
    print(f"{'='*60}")
    print(cmd)
    if not args.dry_run:
        subprocess.run(cmd, shell=True, check=False)

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", "-e", required=True,
                    help='experiment name, e.g. "cf_stabilization_sweep"')
    ap.add_argument("--debug", action="store_true",
                    help="If set, run in debug mode with only first experiment")
    ap.add_argument("--dry_run", action="store_true",
                    help="If set, only print the commands without running them")
    ap.add_argument("--skip_experiments", type=int, nargs="+", default=[],
                    help="List of experiment indices (1-based) to skip, e.g. --skip_experiments 1 3 5")
    args = ap.parse_args()

    jobs = get_experiment_commands(args)
    print(f"Total jobs: {len(jobs)}")

    if args.debug:
        jobs = jobs[:1]
    
    # Filter out skipped experiments
    skip_indices = set(idx - 1 for idx in args.skip_experiments if 1 <= idx <= len(jobs))
    if skip_indices:
        print(f"Skipping experiments: {sorted(i + 1 for i in skip_indices)}")
        jobs = [cfg for i, cfg in enumerate(jobs) if i not in skip_indices]
        print(f"Running {len(jobs)} experiments")

    for i, cfg in enumerate(jobs, 1):
        run_job(args, cfg, i, len(jobs))

if __name__ == "__main__":
    start_time = time.time()
    main()
    runtime = (time.time() - start_time) / 3600
    print(f"\nRuntime for all experiments: {runtime:.2f} hours")
