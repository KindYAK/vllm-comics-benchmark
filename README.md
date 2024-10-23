# LLM Comics Benchmark

Welcome to the **LLM Comics Benchmark**! We're on a quest to evaluate Large Language Models (LLMs) with visual capabilities using the most intellectually rigorous content available: web comics. Because if an AI can't appreciate a good comic strip, what's the point?

## Overview

The overall flow of the project:
- **Break down web comics into individual panels** using [Kumiko](https://github.com/njean42/kumiko).
- **Evaluate LLMs** (like GPT-4o and GPT-4o-mini) on their ability to understand and sequence shuffled panels back into the correct order.
- **Compute metrics** to quantify their performance.
- **Provide a framework** for adding more comics and models in the future.

## Installation 
Clone the repository and install dependencies with Poetry:

```bash
git clone https://github.com/KindYAK/vllm-comics-benchmark.git
cd vllm-comics-benchmark
poetry install
```

## Usage
**TODO**

```Scripts to run evaluations locally are coming soon.```


## Metrics
We use an average of the **Longest Common Subsequence (LCS) Length** and the **Bubble Sort Distance** to evaluate model performance. The LCS metric is effective for assessing how well an LLM can identify large correct chunks of the plot, especially in comics with many panels. It focuses on the longest sequence of panels that are correctly ordered. On the other hand, the Bubble Sort Distance measures the minimal number of adjacent swaps needed to correct the model's panel sequence, which is better for evaluating smaller misplacements and overall sequence similarity. Also, random baseline for this approach is ~0.5, which makes the proposed metric more interpretable.

### Results
| Model             | XKCD loss | Average |
|-------------------|-----------|---------|
| Random (baseline) | 0.52      | 0.52    |
| GPT-4o-mini       | 0.57      | 0.57    |
| GPT-4o            | 0.41      | 0.41    |
| **Average**       |           |         |
*More comics and models coming soon!*

## Results Format

The evaluation results are stored in JSON files, one for each model (e.g., `results-xkcd-gpt-4o.json`, `results-xkcd-gpt-4o-mini.json`). Each JSON file contains a list of dictionaries, where each dictionary corresponds to a single comic evaluation.

### Example Entry

```json
[
  {
    "panels_path": "data/xkcd/736-Cemetery.jpg",
    "imgs_original": ["panel_1.jpg", "panel_0.jpg"],
    "imgs_predicted": ["panel_1.jpg", "panel_0.jpg"],
    "correct_order": [1, 0],
    "predicted_order": [0, 1],
    "reasoning": "...",
    "failed": false
  },
  ...
]
```
### Field Descriptions

- panels_path: The file path to the original comic image.
- imgs_original: A list of filenames for the extracted comic panels in their original (correct) order.
- imgs_predicted: A list of filenames for the panels as predicted by the model.
- correct_order: A list of indices representing the correct order of the panels.
- predicted_order: A list of indices representing the panel order as predicted by the model.
- reasoning: The model's reasoning or explanation for the predicted panel order.
- failed: A boolean indicating whether the model failed to output the correct indices after multiple attempts (e.g., 5 retries). If true, the predicted_order is automatically corrected.


# Data
Currently, we are using comics from :
- [XKCD](https://xkcd.com/), which provides open licenses for use.
- **TODO** More web comics with open licenses to be added soon.

- **TODO**: Script to download data automatically.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request.


## License
[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/) 

You are free to share and adapt the material as long as appropriate credit is given.
