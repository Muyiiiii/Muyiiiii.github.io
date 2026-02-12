---
title: "Other Thoughts"
date: 2026-02-05
---
Some of Other Thoughts

# Agentic RL 思考

感觉RL只是在规范LLM的回答方式，并没有增强LLM的智能

* GRPO就是只增强了比均值表现好的action，但是感觉没有新的知识注入到模型的paremeter中。或者说action的规范也是一种对于知识的梳理？


# NLP 模型处理方法变化

* N-gram: 计算相连词汇的概率，但不知语意
* word2vec：用中心词预测来学习语意，但是窗口固定，只能看n-1个词汇
* RNN：循环网络，能处理任意长度，但是长时记忆容易忘，而且必须顺序执行，很慢
* LSTM：解决了长时记忆，但还是顺序执行
* Attention：并行，且能看全部

# LSTM

![lstm-3.svg](../files/blog_assets/lstm-3.svg)

Link: [9.2. 长短期记忆网络（LSTM） — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)

H是短时记忆，C是长时记忆

用sigmod来决定是不是计算重要性，来遗忘或者记忆

用tanh来得到embedding
