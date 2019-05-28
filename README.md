# PAN-18

Data and code for the evaluation of the PAN18 shared task in authorship masking (http://ceur-ws.org/Vol-2125/invited_paper_16.pdf).
The folder `obfuscation_evaluation` contains the “real” data, the two folders `mock1_datasets` and `mock2_datasets` contain
the original data and some additional fake data used for the robustness experiments.

## Use cases
* compute the total score of an obfuscator for a specific dataset (using the available verifiers on that specific dataset). e.g. this computes the total score of
obfuscator A on the PAN13 dataset:
```bash
python3 safety_evaluation.py obfuscation_evaluation/pan13-test-dataset obfuscation_evaluation/pan13-test-dataset-truth.txt --obfuscation obfuscation_evaluation/pan13-test-dataset-obfuscationA
```
(in the example, additional warnings are printed to stderr because some verifier did not report results on all test problems;
these can be surpressed by passing the flag `--silent-on-missing-values`).

* summarize the outputs of all verifiers for a specific dataset, their effectiveness and coverage scores. e.g. this writes a summary for PAN13 to the
file `pan13_setup_summary.html`:
```bash
python3 safety_evaluation.py obfuscation_evaluation/pan13-test-dataset obfuscation_evaluation/pan13-test-dataset-truth.txt --write-setup-html-overview pan13_setup_summary.html
```

* summarizes how an obfuscator flips the decisions of all verifiers for a specific dataset. e.g. this writes a summary how obfuscator A changes decisions on PAN13:
```bash
python3 safety_evaluation.py obfuscation_evaluation/pan13-test-dataset obfuscation_evaluation/pan13-test-dataset-truth.txt --obfuscation obfuscation_evaluation/pan13-test-dataset-obfuscationA --write-obfuscation-html-overview pan13_obfuscationA_summary.html
```
(this also outputs the total score of obfuscator A on PAN13; this can be suppressed by passing the flag `--suppress_final_score`)

* summarize the confidence scores of all verifiers for a specific dataset (shows boxplots)
```bash
python3 safety_evaluation.py obfuscation_evaluation/pan13-test-dataset obfuscation_evaluation/pan13-test-dataset-truth.txt --show-verifier-distributions
```

* print the maximal achievable score on a dataset (for dirty implementation reasons, needs an obfuscation to be passed)
```bash
python3 safety_evaluation.py obfuscation_evaluation/pan13-test-dataset obfuscation_evaluation/pan13-test-dataset-truth.txt --obfuscation obfuscation_evaluation/pan13-test-dataset-obfuscationA --max-score
```

* give more details on the performance of an obfuscator
```bash
python3 safety_evaluation.py obfuscation_evaluation/pan13-test-dataset obfuscation_evaluation/pan13-test-dataset-truth.txt --obfuscation obfuscation_evaluation/pan13-test-dataset-obfuscationA --show-more-statistics
```
