# QA Bot

Project on question answering, semantic search and BERT, using the `transformers` and `tensorflow-hub` libraries.

## Resources

Read or watch:

* [Question Answering (Q&A) Systems with LLMs](https://intranet-dlh.hbtn.io/rltoken/IfUPZkhvQm61iOsf1PPGvQ)
* [Improving Language Understanding by Generative Pre-Training (2018)](https://intranet-dlh.hbtn.io/rltoken/vFBWBmL-idNfApsMEt-RRw)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://intranet-dlh.hbtn.io/rltoken/mIab2UZnm64p6TSa-xPTrA)
* [SQuAD 2.0](https://intranet-dlh.hbtn.io/rltoken/vigSZvwAnpAoqGUHBQ6TIw)
* [Know What You Don't Know: Unanswerable Questions for SQuAD (2018)](https://intranet-dlh.hbtn.io/rltoken/ASnUmAeZLCy4fuIlZQocIg)
* [GLUE Benchmark](https://intranet-dlh.hbtn.io/rltoken/KDjvnz6EKqTiRjQwhQ47tQ)
* [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding (2019)](https://intranet-dlh.hbtn.io/rltoken/zH4c1olq3tnR3DFd-ZA8fQ)
* [Speech-transformer: A no-recurrence sequence-to-sequence model for speech recognition (2018)](https://intranet-dlh.hbtn.io/rltoken/lAfZfIU9arRF-S_egdfrJw)

More recent papers in NLP:

* [Generating Long Sequences with Sparse Transformers (2019)](https://intranet-dlh.hbtn.io/rltoken/PK38qBHz6DSLNpmZCds6dg)
* [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (2019)](https://intranet-dlh.hbtn.io/rltoken/XR2NKelmGGXwwaIsLOMK_g)
* [XLNet: Generalized Autoregressive Pretraining for Language Understanding (2019)](https://intranet-dlh.hbtn.io/rltoken/EUM1GHB5x2mIsRqeAcTdqg)
* [Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](https://intranet-dlh.hbtn.io/rltoken/E6AzWVl0M01VKZGmNbQsDw)
* [Language Models are Few-Shot Learners (GPT-3, 2020)](https://intranet-dlh.hbtn.io/rltoken/FRt-dFnVqYwF2p1FzXGG2w)
* [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations (2020)](https://intranet-dlh.hbtn.io/rltoken/gPHPLI60U1hS4AJ1Uht67A)
* [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://intranet-dlh.hbtn.io/rltoken/rKhEDzg0RMDVz884raV-rA)
* [Titans: Learning to Memorize at Test Time (2024)](https://intranet-dlh.hbtn.io/rltoken/Xln_GqTXtlweAb3Mi5z8Kw)

To keep up with the newest papers and their code bases go to [paperswithcode.com](https://intranet-dlh.hbtn.io/rltoken/pKN9GRieRoP-X6PkA8MwGA). For example, check out the [ranked list of state of the art models for Language Modelling on Penn Treebank](https://intranet-dlh.hbtn.io/rltoken/qIB-iXA--vsb7rAl3UyQIg).

## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](https://intranet-dlh.hbtn.io/rltoken/WzIBN64E6sbdFRApZ4_hLg), without the help of Google:

* What is Question-Answering?
* What is Semantic Search?
* What is BERT?
* How to develop a QA chatbot
* How to use the `transformers` library
* How to use the `tensorflow-hub` library

## Requirements

- Python 3.9
- numpy 1.25.2
- tensorflow 2.15
- tensorflow-hub 0.15.0
- transformers 4.44.2
- pycodestyle 2.11.1

## Files

| File | Description |
|------|-------------|
| [0-qa.py](./0-qa.py) | `question_answer(question, reference)` finds a snippet of text within a reference document to answer a question using BERT, returns `None` if no answer is found |
