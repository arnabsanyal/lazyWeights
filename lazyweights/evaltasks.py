"""Downstream tasks with lm-eval-harness on the in-memory (dequantized) module: few-shot counts and the metric read
per task. gsm8k_cot is 8-shot chain of thought (strict and flexible answer extraction)."""
import os
import time

TASKS = {"hellaswag": 0, "arc_easy": 0, "arc_challenge": 0, "piqa": 0, "winogrande": 0, "mmlu": 5, "gsm8k": 5,
         "gsm8k_cot": 8, "gsm8k_cot_dev": 8}

# dev tasks for choosing among operating points without the test sets: base task, split, first n documents
DEV = {"gsm8k_cot_dev": ("gsm8k/gsm8k-cot.yaml", "train", 1000)}


def tasks(model, tok, names, batch):
    import tempfile
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch, max_length=2048)
    tm = None
    if any(t in DEV for t in names):
        d = tempfile.mkdtemp()
        for t, (base, split, _) in DEV.items():
            with open(os.path.join(d, t + ".yaml"), "w") as f:
                f.write("include: %s\ntask: %s\ntest_split: %s\n"
                        % (os.path.join(os.path.dirname(lm_eval.__file__), "tasks", base), t, split))
        tm = TaskManager(include_path=d)
    res = {}
    for t in names:
        t1 = time.time()
        r = lm_eval.simple_evaluate(model=lm, tasks=[t], num_fewshot=TASKS[t], log_samples=False, random_seed=0,
                                    numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0,
                                    task_manager=tm, limit=DEV[t][2] if t in DEV else None)
        m = r["results"][t]
        key = {"gsm8k": "exact_match,strict-match", "gsm8k_cot": "exact_match,strict-match",
               "gsm8k_cot_dev": "exact_match,strict-match", "piqa": "acc,none", "winogrande": "acc,none",
               "mmlu": "acc,none"}.get(t, "acc_norm,none")
        res[t] = float(m[key])
        res[t + "_stderr"] = float(m[key.replace(",", "_stderr,")])
        if t.startswith("gsm8k_cot"):   # also the answer extracted anywhere in the generation
            res[t + "_flex"] = float(m["exact_match,flexible-extract"])
        print("%s %.4f (%.0fs)" % (t, res[t], time.time() - t1), flush=True)
    return res
