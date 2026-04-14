# grammar_search
Code for the paper "Grammar Search for Multi-Agent Systems"

Example command:
```
python -m run_grammar_search \
    --iterations 30 \
    --dataset math \
    --forced-iterations 30 \
    --forced-max-length 6 \
    --forced-sampling random \
    --beta-top-n 100 \
    --credit full \
    --alpha-init 1.7 \
    --beta-init 0.3 \
    --problems 160 \
    --runs 5 \
    --test-runs 8 \
    --top-systems 1 \
    --output data/math_bbgpt4o_ogpt41mini_forced_30_iters_results.json \
    --log-file /tmp/grammar_search_data/math_bbgpt4o_ogpt41mini_forced_30_iters_logs.log \
    --debug-dir /tmp/grammar_search_data/math_bbgpt4o_ogpt41mini_forced_30_iters_logs \
    --cache-dir cache_dir
```

This command runs 30 random sampling iterations of grammar search for the math dataset.
