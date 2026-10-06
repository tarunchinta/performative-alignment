# Responsible NLP Research Checklist

Paper: *Performative Alignment: Preference Learning under Deployment-Dependent Utility* (v7, camera-ready)

## A. For all submissions

**A1. Did you describe the limitations of your work?**  
Yes. Unnumbered **Limitations** section after Conclusion (does not count toward page limit).

**A2. Did you discuss any potential risks of your work?**  
Yes. **Ethics Statement** (after Limitations): homogenization risk, LLM-judge bias, overclaiming on cultural impacts; Section 5.1 scopes homogenization as predicted signature only.

## B. Did you use or create scientific artifacts?

**B1. Did you cite the creators of artifacts you used?**  
Yes. Section 2 and References cite RLHF, VPL, performative prediction, and corpus-study sources; Section 4 cites gpt-5.4 (Azure AI Foundry).

**B2. Did you discuss the license or terms for use and/or distribution of any artifacts?**  
Yes. Code, configurations, cached judge responses, and result files at https://github.com/tarunchinta/performative-alignment, released under the MIT License; gpt-5.4 accessed via Azure API under provider terms. Menu texts are author-written for this study.

**B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use?**  
Yes. gpt-5.4 used only for pairwise preference elicitation via chat-completions API; no fine-tuning or redistribution of model weights.

**B4. Did you discuss the steps taken to check whether the data that was collected/used contains any information that names or uniquely identifies individual people or offensive content?**  
Yes. Ethics Statement: no human participants or PII; menu texts are synthetic one-line city descriptions authored for the experiment.

**B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages, and demographic groups?**  
Yes. Section 4.6–4.7: 100-item English menu, 10 style clusters, single domain (city-at-night one-liners), one judge family (gpt-5.4); Limitations lists scope constraints.

**B6. Did you report relevant statistics like the number of examples, details of train/validation/test splits, etc.?**  
Yes. Section 4: 100 items, 10 clusters, λ sweep, 10 seeds (Exp 1–2), 3 pair-sampling seeds (Exp 3 Phase 3), probe-cluster design, gate thresholds.

## C. Did you run computational experiments?

**C1. Did you report the number of parameters in the models used, the total computational budget, and computing infrastructure used?**  
Partial. No trainable LLM parameters (frozen judge + NumPy/SciPy optimization). Compute reported in Section 4.7: 247,800 logged judge responses (248,070 API requests including retries); 109.95M input / 1.24M output / 111.19M total tokens; 448 tokens/request avg; Azure estimated cost US$0 for queried interval; Exp 1–2 on local CPU.

**C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values?**  
Yes. Section 4.1 (λ sweep), 4.6–4.7 (τ=0.07 quantile target, temperature=1 API constraint, 5 repeats, bootstrap CIs, surrogate form selection via held-out log-likelihood).

**C3. Did you report descriptive statistics about your results?**  
Yes. Mean ± sd over seeds; bootstrap CIs for psychometrics; tables in Sections 4.3, 4.7.

**C4. If you used existing packages, did you report their implementation, model, and parameter settings?**  
Yes. Section 4.6 implementation notes; repository documents Python dependencies and run scripts.

## D. Did you use human annotators?

**D1–D4.** N/A. No human annotators; Experiment 3 uses an LLM judge. Ethics Statement discusses judge-as-rater limitations.

## E. Did you use AI assistants?

**E1. Did you include information about your use of AI assistants?**  
Yes. Ethics Statement: generative AI assisted language revision, code development/debugging, and exploratory literature search; author verified literature, reviewed code/text, executed experiments, and takes responsibility.
