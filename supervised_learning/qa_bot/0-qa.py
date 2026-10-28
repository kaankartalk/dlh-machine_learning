#!/usr/bin/env python3
"""Module that finds a snippet of text to answer a question using BERT"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question

    question: string containing the question to answer
    reference: string containing the reference document

    Returns: string containing the answer, or None if no answer is found
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    q_tokens = tokenizer.tokenize(question)
    r_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + q_tokens + ['[SEP]'] + r_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = ([0] * (1 + len(q_tokens) + 1) +
                      [1] * (len(r_tokens) + 1))

    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])

    start = int(tf.argmax(outputs[0][0][1:]) + 1)
    end = int(tf.argmax(outputs[1][0][1:]) + 1)

    answer_tokens = tokens[start:end + 1]
    if end < start or not answer_tokens:
        return None

    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if not answer.strip() or answer.strip() in ('[CLS]', '[SEP]'):
        return None
    return answer
