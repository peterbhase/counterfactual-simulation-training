# Verbalization Rewrite Messages (VFT-Style)

**Function:** `build_verbalization_rewrite_messages()` in `utils.py`

**Purpose:** Rewrite unbiased reasoning to faithfully verbalize bias influence (Verbalized Faithfulness Training style).

---

## Prompt Template

**System:**
```
In this task, we are biasing models towards particular answers with the following bias: {bias_description}
Since it is impractical to assume that models will never be biased, we want to come up with examples how models can be affected by cues in the prompt and make this transparent to the user.
We will give you unbiased reasoning for a new question, and you will need to give the corresponding faithful biased response.

- Faithful biased reasoning gives the biased answer AND verbalizes the fact that it is being influenced by the features of the prompt.
- Use the unbiased reasoning as a template. Make minimal edits to this reasoning, where you attribute giving the biased answer to the bias.
- Do not include any other text.
- Don't say you are ambiguously 'affected' by the bias, say why it's influencing your answer.
- Make sure that no text comes after giving the final answer.
{formatting_guidelines}

Format instructions: Place your rewritten reasoning in <rewritten_reasoning> tags, followed by the answer supported by this reasoning in <rewritten_answer> tags. Do not think out loud before writing the new reasoning, just write it directly, followed by the answer reached by that reasoning. Your output should follow the format: <rewritten_reasoning>...</rewritten_reasoning> <rewritten_answer>...</rewritten_answer>.
```

**User:**
```
I will give you the relevant information below.

<original_question>{original_question}</original_question>

<original_reasoning>{original_reasoning}</original_reasoning>

<original_model_answer>{original_model_answer}</original_model_answer>
```

**Assistant:**
```
<rewritten_reasoning>{faithful_biased_reasoning}</rewritten_reasoning>

<rewritten_answer>{biased_answer}</rewritten_answer>
```

---

## Variables

| Variable | Description |
|----------|-------------|
| `{bias_description}` | Description of the bias cue (from `bias_description_dict`) |
| `{formatting_guidelines}` | 3 randomly selected formatting tips (see below) |
| `{original_question}` | Original MC question (with bias cue) |
| `{original_reasoning}` | Unbiased reasoning to use as template |
| `{original_model_answer}` | Model's original answer |

---

## Formatting Guidelines Pool

3 of these are randomly sampled per prompt for `{formatting_guidelines}`:

- In some cases the bias will be toward the correct answer so in some cases briefly consider if the biased answer seems plausible.
- You can mention the feature in the middle of the reasoning as well, not just at the end.
- Mention the bias in a natural way as if you were encountering this in the wild.
- Make the mention of the bias succinct and subtle.
- Don't use the word 'bias' or 'biased' to describe it, come up with a natural sounding way to mention it. Other words: prompt, format, input, feature, indicator, shortcut, hint, cue, etc.
- Match the style of the unbiased reasoning.
- Merely note the presence of the feature at first, only later consider whether it might be worth using in the reasoning.

---

## Key Differences from `build_rewrite_messages`

| Aspect | `build_rewrite_messages` | `build_verbalization_rewrite_messages` |
|--------|-------------------------|---------------------------------------|
| **Counterfactual info** | Yes (CF question, reasoning, answer) | No |
| **System prompt** | Fixed template | Per-datapoint (random guidelines) |
| **Goal** | Make reasoning predictable across orig/CF | Verbalize bias influence explicitly |
| **Style** | Detailed rewriting instructions | Minimal edits to unbiased template |
