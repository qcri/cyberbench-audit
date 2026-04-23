1- [mohannad]: update benchmark and model selection criteria.
2- [cathrine]: add missing models (reasoning)
3- [cathrine]: update script to get benchmarks from their original repos
4- [cathrine]: exp #1: baseline no changes to benchs
  - run each benchmark (their original dataset + inference prompt and its params + answer eval script) against each model.
  - if the benchmark is missing the inference prompt or eval script, use our default inference prompt (same for all models) or eval script (judge model).
  - convert all outputs into one unified format.
  - expected output: a table of benchmarks vs models (baseline eval).
5- [ayemen]: exp #2: evaluating original eval scripts
  - use the output from exp #1 to check if the key answers for each question in each benchmark is correct.
  - to do this, we're using a majority voting by x judges (the voting threshold varies from 0.25 to 0.75) to flag possibly wrong key ansawers.
  - expected output:
    - a figure showing #flagged vs threshold for each benchmark.
    - benchmarks that have more than one task will have multiple lines in the figure.
    - concentrated long tail distro that plataus after some threshold
    - table of benchmarks vs # flagged and % flagged
    - validation of classification based on random sampling and manual labeling [ask everyone to help]
6- [] exp #3: evaluating how model-specific inference prompt affects (one) benchmark results
7- [] exp #4: evaluating how language affects benchmarking results (translate en to ar and evaluate)
8- [] exp #5: 
