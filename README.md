# Adversarial Attack on Open Source Language Models

## Overview
This repository demonstrates an embedding space attack on a language model, specifically GPT-Neo 1.3B. The attack manipulates the model's response by subtly altering the input embeddings. The goal is to understand how the responses of a language model can be manipulated, providing insights into potential vulnerabilities and defense mechanisms in AI systems.

## Key Resources
1. [Medium Article](https://saksheepatil05.medium.com/cracking-the-code-how-adversarial-attacks-manipulate-ai-language-models-239620395e58) - Cracking the Code: How Adversarial Attacks Manipulate AI Language Models

2. [Original Paper](https://arxiv.org/abs/2310.19737) - Adversarial Attacks and Defenses in Large Language Models: Old and New Threats by Leo Schwinn, David Dobre, Stephan Günnemann, and Gauthier Gidel

3. [Original Paper Repository](https://github.com/SchwinnL/LLM_Embedding_Attack) 

## Files

```check_torch.py```: This script checks the hardware settings and configurations of your PyTorch installation. It helps ensure that your system is properly set up to run the model and the attack scripts.

```embedding_attack.py```: The core script that runs the embedding space attack on the GPT-Neo 1.3B model. This file contains the implementation of the attack, including generating adversarial examples by manipulating embeddings to achieve the target output.

```establish_baseline.py```: 
This script establishes a baseline for the model's responses. It runs the model on a predefined input prompt and prints the output. This helps in comparing the baseline output with the manipulated outputs during the attack.


# Getting Started
## Prerequisites
- Python 3.7 or later
- PyTorch 1.9 or later
- Transformers library from Hugging Face

<b>Install the required libraries using:</b>

```bash
pip install -r requirements.txt
```

## Running the Scripts

<b>Check Hardware Settings:</b>

```bash
python check_torch.py
```

<b>Establish Baseline Response:</b>

```bash
python establish_baseline.py
```

<b>Run the Embedding Space Attack:</b>

```bash
python embedding_attack.py
```
