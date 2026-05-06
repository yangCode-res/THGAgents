# THGAgents

This is the official repository for the IJCAI 2026 paper:

**THGAgents: Traceable Biomedical Hypothesis Generation via Dynamic Causal Reasoning**

## Overview

THGAgents is a framework for traceable biomedical hypothesis generation. It aims to support biomedical discovery by generating hypotheses through dynamic causal reasoning while providing interpretable reasoning traces.

This repository will provide the official implementation, datasets, experimental settings, and instructions for reproducing the results reported in the paper.

## Paper

**Title:** THGAgents: Traceable Biomedical Hypothesis Generation via Dynamic Causal Reasoning  
**Conference:** International Joint Conference on Artificial Intelligence, IJCAI 2026

## Repository Status

The code and documentation are being organized and will be released soon.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{yang2026thgagents,
  title     = {THGAgents: Traceable Biomedical Hypothesis Generation via Dynamic Causal Reasoning},
  author    = {Yang, Mingjian and Dong, Kun and Liu, Juan and Lim, Kevin},
  booktitle = {Proceedings of the 35th International Joint Conference on Artificial Intelligence},
  year      = {2026}
}


### API Keys Setup

Create a `.env` file in the project root directory:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE_URL=  # or your custom endpoint
OPENAI_MODEL=gpt-4o  # or gpt-4, gpt-3.5-turbo, etc.

# Optional: Mistral OCR API (for PDF processing)
MISTRAL_API_KEY=your_mistral_api_key_here
```

**Important**: Never commit your `.env` file or API keys to version control.

## Usage

### Run Full Benchmark Pipeline

Execute the main script to run the complete benchmark pipeline:

```bash
python run.py
```






