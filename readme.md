# Text Classification Model Comparisons
This repository showcases the research of "Bigger Isn't Always Better: A Benchmark of Language Models vs. Embedding Methods for Text Classification".

## Code Owners
[Nathaniel Maymon](nathanjm91@gmail.com)

## Overview

<p>
    The research was done as part of a Masters program in Information Systems & Data Science at The Academic College Tel-Aviv Yafo, school of Information Systems.
</p>

<p>
   The research compares a total of 50 models across two datasets (25 per model).<br/>
   The goal of the research is to examine the tradeoffs between model types and sizes for text calssification tasks, mainly in the aspect of overall accuracy and computational costs.
</p>

<p>
    Thus, frameworks like <code>Unsloth</code> with <code>QLoRA</code> and smaller models such as MiniLM, BERT, Distill-RoBERTa, XLM-RoBERTa etc. were used.
</p>

The main models examined were:
* Traditional ML & DL models with the combination of PLM based embeddings and NLP derived features as inputs.
* Small Language Models: Distill-RoBERTa, BERT, LLaMA 3.2 1B & 3B etc.
* Large Language Models: GPT-4o.

Finally, all models were compared against one another.

### Main Process
1) EDA, feature engineering and preprocessing of both datasetes. See notebooks `Sales Data Text Classification`, and `IMDB Data Text Classification`.
2) Train, test and evaluate models per each dataset. See all notebooks under `models` directory. 
3) Compare all model metrics across both datasets. See the notebook `Text Classification Model Comparison`.

### Research Conclusions
* LLMs and SLMs are more versatile and easier on inputs\ouputs:
    * General Models enable nearly any text input as is and almost any result simply with prompt engineering.
    * They tend to under perform on complex case-specific text classification tasks.
    * Fine-tuning models almost always provides and improvement in results.
    * Have the highest deployment costs: Privacy, latency, computational resources, financiel etc. even when using QLoRA and Unsloth.
* PLM based embedding approaches with downstream traditional ML & DL models are highly effective methods for text classification:
    * Siginificant savings on computational resources for both training & inference, including training time.
    * Higher accuracy than LLMs in most cases.
    * Some only require a GPU for the embedding pahse.
    * Abillity to add engineered featured for additional context.
    * Cons: Requires a solid data preprocessing pipeline and is task specific.
* Fine-tuned PLMs are best value overall:
    * Can handle the inputs as is, i.e no special pipeline required.
    * Low computational resoureces required and relativly small amount of time required for fine-tuning.
    * High classification accuracy rates on average.
    * Best of both worlds.
    * Cons: Requires fine-tuning, and is task-specific.

<b>Bottom Line</b>: 
* If you have a very general use case that requires a large knowledge base, use an LLM\SLM for text classification. 
* However, if you have a custom use case, using an SLM\PLM based embedding approach with traditinal ML\DL models will give you the best "bang for your buck" by far. 
* Finally, While using a model like Distill-RoBERTa is easiset on input, fine-tuning, and inference while on average providing the best accuracy.

## Repo overview
repo layout explanation

* `utils`: Files containing utility functions for various operations throughout the notebooks.
* `datasets`: Contains the base datasets (as described below), and is where all the intermediate datasets from the notebooks are saved.
* `models`: Main directory containing a subdirectories for each model's notebooks and model-specific utils files. Go there to view the model runs specifically.

## Datasets
The datasets used are:

1) IMDB Movie Review Dataset
```@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

2) A propriatry sales form inquiries dataset of an online SaaS company. A sample of the a cleaned dataset is available in the repository.

## Software
```
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {http://github.com/unslothai/unsloth},
  year = {2023}
}
```

## How to Run
* <b>System Requirements</b>
    * All DL models require an NVIDIA GPU.
    * Most DL models require up to 8GB VRAM. Though Some require up to ~22GB of GPU VRAM.
    * At least 16GB of system RAM.
    * The research itself was conducted on a Intel x86 based system and Google Colab Pro. If running on Mac, adaptations migtht be required.
* <b>Software Requirements</b>
    * Python 3.11.5
    * Huggingface API keys & tokens
    * OpenAI API Key (if runnig the model)
    * CUDA installation
    * Unsloth
    * Jupyter Notebook \ Jupyter Lab
    * Virtual Enviorment VENV or Poetry

#### Installations
* Using Poetry or pip in a virtual enviornment, install all packages in `requirements.txt`.<br/>
Best to use Poetry to handle package managment etc. though VENV is also an option.

* Validate that Python can identify your GPU and CUDA.
    * [Download & install CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
    * Make sure all is set up properly.

#### Set Up
* In the file `utils/env.py` insert the path to the project directory on the system. It's required for the notebooks so they can perform local imports and find files in the project.
    * `base_path=r'path_to_your_project'`
* If using the OpenAI model: insert your OpenAI API key
    * `openai_key='your_openai_key'`

#### Running
* Follow the process described in the "Main Process" section.
* Get a cup of coffey, it will take a while to run.
