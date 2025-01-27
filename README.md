# Automatic Prompt Optimization
This work is an adaptation of the original [ProTeGi](https://github.com/microsoft/LMOps/tree/main/prompt_optimization) project by Microsoft. It has been customized to support open-source language models from Hugging Face, utilizing vLLM as the underlying engine.

The primary focus is on fact-checking tasks, but the approach is flexible and can be applied to a wide range of classification tasks. For more details, please refer to the original paper: [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495).


The main program is defined in `main.py`. Given an initial prompt in markdown format, which follows this structure:
```markdown
# Task
You are a climate expert. Your function is to evaluate claims related to climate change found on the internet.

# Output format
Answer "Supports" or "Refutes" as labels, without quotes.

# Prediction
Text: {{ text }} 
Label:
```
the program is capable of optimizing the `# Task` section according to the chosen test set. The program will output a test file that shows the optimization progression, step-by-step.

It accepts the following input parameters:
- `--task` (default: `climate_fever`) corresponds to the name of the task from which to extract the train and test sets. The given task name is parsed by the `get_task_class` method, defined in the main script.
- `--prompts` (default: `prompts/climate_fever.md`) indicates the path to the initial prompt.
- `--out` (default: `test_out.txt`) specifies the path for the output file.
- `--paraphraser` (default: `None`) is the name of the optional model to be used as the WORKER.
- `--max_tokens` (default: `4`) sets the maximum number of tokens for the PREDICTOR.
- `--model` (default: `mistralai/Mistral-7B-Instruct-v0.2`) specifies the model to use as the PREDICTOR.
- `--temperature` (default: `0.8`) sets the temperature for the PREDICTOR. As for the WORKER, its parameters are currently fixed at temperature=0.8 and max_tokens=1024.
- `--rounds` (default: `6`) defines the number of optimization steps.
- `--beam_size` (default: `4`) determines how many "branches" to create at each step.
- `--n_test_exs` (default: `200`) specifies how many records from the test set to use when evaluating the output. If `n_test_exs` > number of records in the test set, then `n_test_exs` will equal the number of test records.

There are additional, more technical parameters that are self-explanatory; please refer to lines `66-85` in the main script.

## Defining a New Task
To define a new task, you just need to edit the `tasks.py` file and add a new class that inherits from `ClassificationTask` (also defined within the file). This class must implement two methods: `get_train_examples` and `get_test_examples`, which respectively return the training and test datasets. These datasets should be lists of maps, structured as follows:

```json
{'id': f'train-{i}', 'label': row['claim_label'], 'text': row['claim']}
```

Finally, the class must implement the `stringify_prediction` method, which simply converts a prediction into its corresponding label.

To better understand this structure, it is recommended to examine and take inspiration from the two existing binary tasks, `PolitifactBinaryTask` and `ClimateBinaryTask`.

## Predictors
The `predictors.py` file defines which engines to use for inference. Currently, the predictor that leverages vLLM is defined as `VLLMPredictor`. It is responsible for generating predictions and also handles the execution of all intermediate optimization steps (see WORKER). It is generalized to accept any open (or closed) model that can run on vLLM. You can specify a different model for both the WORKER and the PREDICTOR.

The class implementing this functionality defines two methods: `eval_multiple` (used by the WORKER) and `inference` (used by the PREDICTOR). If no separate WORKER is specified, the same model as the PREDICTOR will be used.

## Optimizers
The `optimizers.py` file contains the logic of the optimizer, implementing the exact algorithm described in the paper [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495).

## Scorers
In `scorers.py`, two types of scorers are defined and used by the `evaluator` to assess the quality of a prompt. The default scorer is `Cached01Scorer`, which simply checks how many predictions are correct compared to the test set. It works only for binary tasks. Alternatively, `CachedLogLikelihoodScorer` can be used for non-binary tasks.

## Evaluator
Several evaluators are defined in the `evaluators.py` file, allowing for different optimization strategies. The default is `UCBBanditEvaluator`, which utilizes the UCB (Upper Confidence Bound) algorithm for multi-armed bandits to guide the optimization process.

## Example of execution
```bash
python main.py --model="meta-llama/Llama-2-7b-hf" --out="res_llama_hybrid_politi_UCB.txt" --paraphraser="mistralai/Mistral-7B-Instruct-v0.2" --rounds=8 --task="politifacts" --prompts="prompts/politifacts.md"
```

<details>
  <summary>Output</summary>

```
{"task": "politifacts", "prompts": "prompts/politifacts.md", "out": "res_llama_hybrid_politi_UCB.txt", "paraphraser": "mistralai/Mistral-7B-Instruct-v0.2", "max_threads": 1, "max_tokens": 4, "model": "meta-llama/Llama-2-7b-hf", "temperature": 0.8, "optimizer": "nl-gradient", "rounds": 8, "beam_size": 4, "n_test_exs": 200, "minibatch_size": 64, "n_gradients": 4, "errors_per_gradient": 4, "gradients_per_error": 1, "steps_per_gradient": 1, "mc_samples_per_step": 2, "max_expansion_factor": 8, "engine": "chatgpt", "evaluator": "bf", "scorer": "01", "eval_rounds": 8, "eval_prompts_per_round": 8, "samples_per_eval": 32, "c": 1.0, "knn_k": 2, "knn_t": 0.993, "reject_on_errors": false, "eval_budget": 2048}
======== ROUND 0
5.269050598144531e-05
('# Task\nYou are a political expert. Your function is to evaluate veridicity about statements related to politics found on the internet. You will assign one of the following labels to each claim:\n- "Supports" if you think the political fact is true\n- "Refutes" if you think the political fact is fake.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:',)
(1.0,)
f1: [0.54]
accuracy: [0.54]
cf_mat: [array([[76, 24],
       [68, 32]])]
======== ROUND 1
403.33202362060547
('# Task\nYou are a political expert. Your function is to evaluate veridicity about statements related to politics found on the internet. You will assign one of the following labels to each claim:\n- "Supports" if you think the political fact is true\n- "Refutes" if you think the political fact is fake.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n In my role as a fact-checker, I will evaluate the accuracy of political statements and assign them the labels "Supports" or "Refutes", depending on the evidence presented.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n As a knowledgeable analyst in political affairs, it is my duty to verify the authenticity of statements concerning politics encountered online. Based on the context and the credibility of the information\'s origin, I will assign the following classification to each claim:\n- "Corroborates" if the political data aligns with the facts\n- "Contradicts" if the political data contradicts the existing knowledge.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\nYou are a political expert. Your function is to evaluate the factual accuracy of statements related to politics found on the internet. You will assign a label that reflects the degree of factual accuracy of the statement. The labels are:\n- "True": the statement is entirely factual and accurate.\n- "Partially true": the statement contains some factual information, but may omit important context or qualifications, or may present distorted information.\n- "False": the statement is not factual and contains incorrect information.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:')
(0.57, 0.55, 0.545, 0.535)
f1: [0.605, 0.555, 0.525, 0.535]
accuracy: [0.605, 0.555, 0.525, 0.535]
cf_mat: [array([[86, 14],
       [65, 35]]), array([[72, 28],
       [61, 39]]), array([[84, 16],
       [79, 21]]), array([[94,  6],
       [87, 13]])]
======== ROUND 2
957.6428484916687
('# Task\n  As an experienced political analyst, it falls upon me to scrutinize political declarations encountered in the digital realm for their veracity. By evaluating the consistency of the data with established facts and the trustworthiness of the statement\'s origin, I can assign an appropriate label to each assertion:\n   - "Validation" if the political information harmonizes with acknowledged truths from a reputable provider,\n   - "Refutation" if the political information clashes with previously authenticated facts from a reliable source,\n   - "Questionable" if the political information might correspond to facts but originates from a dubious or doubtful origin.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\nYou are a political fact-checking expert. Your function is to evaluate the veracity of political statements based on publicly available information. Follow these guidelines to make your assessment:\n- Research the statement using credible sources, such as government websites, reputable news outlets, and scholarly articles.\n- Consider the context and source of the statement.\n- Fact-check key claims in the statement.\n- Use logical reasoning to evaluate the statement.\n- Determine if the statement is supported by sufficient evidence or if it is a falsehood.\n- Use the following labels to denote your assessment:\n  * "Supports" if you find that the statement is true based on your research.\n  * "Refutes" if you find that the statement is false.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n In my role as a fact-checker, I will evaluate the accuracy of political statements and assign them the labels "Supports" or "Refutes", depending on the evidence presented.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n As a political analyst, it is my responsibility to assess the accuracy of political statements circulating on the web. I will assign one of two labels to each assertion: "True" if the information aligns with proven facts, or "False" if it contradicts the known political truth.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:')
(0.5535714285714286, 0.5178571428571429, 0.5178571428571429, 0.5178571428571429)
f1: [0.545, 0.575, 0.59, 0.53]
accuracy: [0.545, 0.575, 0.59, 0.53]
cf_mat: [array([[89, 11],
       [80, 20]]), array([[72, 28],
       [57, 43]]), array([[82, 18],
       [64, 36]]), array([[92,  8],
       [86, 14]])]
======== ROUND 3
931.2800014019012
('# Task\n In my role as a fact-checker, I will evaluate the accuracy of political statements and assign them the labels "Supports" or "Refutes", depending on the evidence presented.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n  Serving as a political commentator, I will evaluate the authenticity of the presented statement. My evaluation will be grounded in the established political data and facts accessible to me at the moment. I will award one of the following ratings to each statement: "Authentic" for those that conform to established facts, or "Misleading" for those that diverge from the authentic political narrative.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n  As an experienced political analyst, it falls upon me to scrutinize political declarations encountered in the digital realm for their veracity. By evaluating the consistency of the data with established facts and the trustworthiness of the statement\'s origin, I can assign an appropriate label to each assertion:\n   - "Validation" if the political information harmonizes with acknowledged truths from a reputable provider,\n   - "Refutation" if the political information clashes with previously authenticated facts from a reliable source,\n   - "Questionable" if the political information might correspond to facts but originates from a dubious or doubtful origin.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\nYou are a political fact-checking expert. Your function is to evaluate the veracity of political statements based on publicly available information. Follow these guidelines to make your assessment:\n- Research the statement using credible sources, such as government websites, reputable news outlets, and scholarly articles.\n- Consider the context and source of the statement.\n- Fact-check key claims in the statement.\n- Use logical reasoning to evaluate the statement.\n- Determine if the statement is supported by sufficient evidence or if it is a falsehood.\n- Use the following labels to denote your assessment:\n  * "Supports" if you find that the statement is true based on your research.\n  * "Refutes" if you find that the statement is false.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:')
(0.5392156862745098, 0.5294117647058824, 0.5294117647058824, 0.5196078431372549)
f1: [0.585, 0.51, 0.52, 0.55]
accuracy: [0.585, 0.51, 0.52, 0.55]
cf_mat: [array([[75, 25],
       [58, 42]]), array([[89, 11],
       [87, 13]]), array([[86, 14],
       [82, 18]]), array([[71, 29],
       [61, 39]])]
======== ROUND 4
936.8699862957001
('# Task\nYou are a political fact-checking expert. Your function is to evaluate the veracity of political statements based on publicly available information. Follow these guidelines to make your assessment:\n- Research the statement using credible sources, such as government websites, reputable news outlets, and scholarly articles.\n- Consider the context and source of the statement.\n- Fact-check key claims in the statement.\n- Use logical reasoning to evaluate the statement.\n- Determine if the statement is supported by sufficient evidence or if it is a falsehood.\n- Use the following labels to denote your assessment:\n  * "Supports" if you find that the statement is true based on your research.\n  * "Refutes" if you find that the statement is false.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n  As an experienced political analyst, it falls upon me to scrutinize political declarations encountered in the digital realm for their veracity. By evaluating the consistency of the data with established facts and the trustworthiness of the statement\'s origin, I can assign an appropriate label to each assertion:\n   - "Validation" if the political information harmonizes with acknowledged truths from a reputable provider,\n   - "Refutation" if the political information clashes with previously authenticated facts from a reliable source,\n   - "Questionable" if the political information might correspond to facts but originates from a dubious or doubtful origin.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n In my role as a fact-checker, I will evaluate the accuracy of political statements and assign them the labels "Supports" or "Refutes", depending on the evidence presented.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n  In my role as a fact-checker, I am responsible for evaluating the authenticity of given political statements. To accomplish this, I will consult trustworthy and dependable sources to authenticate or debunk the statements. Upon completion of my investigation, if the statement is determined to be accurate, it will be tagged with "Supports," and if it is proven to be incorrect, it will carry the label "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:')
(0.5784313725490197, 0.5392156862745098, 0.5196078431372549, 0.5196078431372549)
f1: [0.61, 0.5, 0.535, 0.55]
accuracy: [0.61, 0.5, 0.535, 0.55]
cf_mat: [array([[85, 15],
       [63, 37]]), array([[79, 21],
       [79, 21]]), array([[72, 28],
       [65, 35]]), array([[71, 29],
       [61, 39]])]
======== ROUND 5
824.2466673851013
('# Task\n   As a trusted fact-checker, my job entails verifying the legitimacy of provided political declarations. To ensure accuracy, I will look up trustworthy and credible resources to authenticate or debunk the statements. By comparing and contrasting the reliable sources\' findings, I aim to tag the statements accordingly. If the majority of sources agree and the evidence from each source aligns, the tag will be "Supports," and if not, the label will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\nYou are a political fact-checking expert. Your function is to evaluate the veracity of political statements based on publicly available information. Follow these guidelines to make your assessment:\n- Research the statement using credible sources, such as government websites, reputable news outlets, and scholarly articles.\n- Consider the context and source of the statement.\n- Fact-check key claims in the statement.\n- Use logical reasoning to evaluate the statement.\n- Determine if the statement is supported by sufficient evidence or if it is a falsehood.\n- Use the following labels to denote your assessment:\n  * "Supports" if you find that the statement is true based on your research.\n  * "Refutes" if you find that the statement is false.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n  As an experienced political analyst, it falls upon me to scrutinize political declarations encountered in the digital realm for their veracity. By evaluating the consistency of the data with established facts and the trustworthiness of the statement\'s origin, I can assign an appropriate label to each assertion:\n   - "Validation" if the political information harmonizes with acknowledged truths from a reputable provider,\n   - "Refutation" if the political information clashes with previously authenticated facts from a reliable source,\n   - "Questionable" if the political information might correspond to facts but originates from a dubious or doubtful origin.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n   In my role as a fact-checker, I am responsible for verifying the authenticity of political declarations. To fulfill this obligation, I will refer to credible and up-to-date publications, those released within the prior month. I will restrict my fact-finding to these publications. If a statement aligns with the facts reported in these sources, it will bear the "Supports" label. Conversely, if a statement conflicts with the facts stated in these sources, it will receive the "Refutes" tag.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:')
(0.5784313725490197, 0.5686274509803921, 0.5588235294117647, 0.5490196078431373)
f1: [0.575, 0.585, 0.57, 0.595]
accuracy: [0.575, 0.585, 0.57, 0.595]
cf_mat: [array([[82, 18],
       [67, 33]]), array([[71, 29],
       [54, 46]]), array([[85, 15],
       [71, 29]]), array([[87, 13],
       [68, 32]])]
======== ROUND 6
967.7805156707764
('# Task\n    In my role as a reliable fact-checker, I\'m responsible for determining the truthfulness of political statements. To accomplish this, I\'ll consult credible and trustworthy sources, characterized by their ability to offer verifiable and dependable data. Such sources may consist of fact-checking organizations, authoritative government websites, well-known news outlets, and prestigious scholarly publications. By carefully examining the information from these sources, I intend to sort the statements into one of two categories: "Supports" or "Refutes." If the information from the majority of sources is consistent and the facts align, the label will be "Supports," and if they don\'t, the tag will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\nYou are a political fact-checking expert. Your function is to evaluate the veracity of political statements based on publicly available information. Follow these guidelines to make your assessment:\n- Research the statement using credible sources, such as government websites, reputable news outlets, and scholarly articles.\n- Consider the context and source of the statement.\n- Fact-check key claims in the statement.\n- Use logical reasoning to evaluate the statement.\n- Determine if the statement is supported by sufficient evidence or if it is a falsehood.\n- Use the following labels to denote your assessment:\n  * "Supports" if you find that the statement is true based on your research.\n  * "Refutes" if you find that the statement is false.\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n   As a trusted fact-checker, my job entails verifying the legitimacy of provided political declarations. To ensure accuracy, I will look up trustworthy and credible resources to authenticate or debunk the statements. By comparing and contrasting the reliable sources\' findings, I aim to tag the statements accordingly. If the majority of sources agree and the evidence from each source aligns, the tag will be "Supports," and if not, the label will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n    Serving as a dependable fact-checker, my role involves confirming the authenticity of political assertions. To guarantee precision, I\'ll consult reliable and reputable sources to authenticate or disprove the claims. By examining the consistent findings from these trustworthy resources, I strive to assign appropriate tags to the statements. If the consensus among sources indicates agreement and the corroborating evidence matches, the tag will be "Supports." Conversely, if the sources disagree or their evidence contradicts, the label will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:')
(0.6176470588235294, 0.5784313725490197, 0.5784313725490197, 0.5686274509803921)
f1: [0.58, 0.555, 0.545, 0.49]
accuracy: [0.58, 0.555, 0.545, 0.49]
cf_mat: [array([[79, 21],
       [63, 37]]), array([[72, 28],
       [61, 39]]), array([[81, 19],
       [72, 28]]), array([[73, 27],
       [75, 25]])]
======== ROUND 7
844.8364734649658
('# Task\n    In my role as a reliable fact-checker, I\'m responsible for determining the truthfulness of political statements. To accomplish this, I\'ll consult credible and trustworthy sources, characterized by their ability to offer verifiable and dependable data. Such sources may consist of fact-checking organizations, authoritative government websites, well-known news outlets, and prestigious scholarly publications. By carefully examining the information from these sources, I intend to sort the statements into one of two categories: "Supports" or "Refutes." If the information from the majority of sources is consistent and the facts align, the label will be "Supports," and if they don\'t, the tag will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n   As a trusted fact-checker, my job entails verifying the legitimacy of provided political statements. To ensure accuracy, I will look up trustworthy and credible resources to authenticate or debunk the statements. By comparing and contrasting the reliable sources\' findings, I aim to tag each statement accordingly. If the majority of sources agree and the evidence from each source aligns, the tag will be "Supports," and if not, the label will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n   As a trusted fact-checker, my job entails verifying the legitimacy of provided political declarations. To ensure accuracy, I will look up trustworthy and credible resources to authenticate or debunk the statements. By comparing and contrasting the reliable sources\' findings, I aim to tag the statements accordingly. If the majority of sources agree and the evidence from each source aligns, the tag will be "Supports," and if not, the label will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n    Serving as a dependable fact-checker, my role involves confirming the authenticity of political assertions. To guarantee precision, I\'ll consult reliable and reputable sources to authenticate or disprove the claims. By examining the consistent findings from these trustworthy resources, I strive to assign appropriate tags to the statements. If the consensus among sources indicates agreement and the corroborating evidence matches, the tag will be "Supports." Conversely, if the sources disagree or their evidence contradicts, the label will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:')
(0.6164383561643836, 0.5753424657534246, 0.5753424657534246, 0.5753424657534246)
f1: [0.6, 0.575, 0.575, 0.575]
accuracy: [0.6, 0.575, 0.575, 0.575]
cf_mat: [array([[81, 19],
       [61, 39]]), array([[84, 16],
       [69, 31]]), array([[82, 18],
       [67, 33]]), array([[84, 16],
       [69, 31]])]
======== ROUND 8
631.4736788272858
('# Task\n   As a trusted fact-checker, my job entails verifying the legitimacy of provided political declarations. To ensure accuracy, I will look up trustworthy and credible resources to authenticate or debunk the statements. By comparing and contrasting the reliable sources\' findings, I aim to tag the statements accordingly. If the majority of sources agree and the evidence from each source aligns, the tag will be "Supports," and if not, the label will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n    In my role as a reliable fact-checker, I\'m responsible for determining the truthfulness of political statements. To accomplish this, I\'ll consult credible and trustworthy sources, characterized by their ability to offer verifiable and dependable data. Such sources may consist of fact-checking organizations, authoritative government websites, well-known news outlets, and prestigious scholarly publications. By carefully examining the information from these sources, I intend to sort the statements into one of two categories: "Supports" or "Refutes." If the information from the majority of sources is consistent and the facts align, the label will be "Supports," and if they don\'t, the tag will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n    As a fact-checker, I am tasked with verifying the truthfulness of statements made in a political context. To accomplish this, I will consult reputable and trustworthy sources, known for their accuracy, unbiased reporting, and ability to provide verifiable and dependable data. Such sources may include non-partisan fact-checking organizations like FactCheck.org and PolitiFact, authoritative government websites, well-established news outlets like The New York Times and The Washington Post, and respected scholarly publications. By carefully examining the information from these sources, I will evaluate the factual accuracy of the statement. If the information from the majority of sources aligns with the statement, the label will be "Supports." If the information from the majority of sources contradicts the statement, the label will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:', '# Task\n   As a trusted fact-checker, my job entails verifying the legitimacy of provided political statements. To ensure accuracy, I will look up trustworthy and credible resources to authenticate or debunk the statements. By comparing and contrasting the reliable sources\' findings, I aim to tag each statement accordingly. If the majority of sources agree and the evidence from each source aligns, the tag will be "Supports," and if not, the label will be "Refutes."\n\n# Output format\nAnswer "Supports" or "Refutes" as labels, without quotes.\n\n# Prediction\nText: {{ text }} Label:')
(0.5862068965517241, 0.5862068965517241, 0.5689655172413793, 0.5517241379310345)
f1: [0.55, 0.525, 0.585, 0.55]
accuracy: [0.55, 0.525, 0.585, 0.55]
cf_mat: [array([[86, 14],
       [76, 24]]), array([[75, 25],
       [70, 30]]), array([[78, 22],
       [61, 39]]), array([[82, 18],
       [72, 28]])]

```

</details>