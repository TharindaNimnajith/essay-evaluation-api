import re
from string import punctuation


def preprocess_text_basic(text):
    text = str(text.strip())
    text = remove_html(text)
    return text


def preprocess_text_advanced(text):
    text = str(text.strip())
    text = remove_html(text)
    text = remove_punctuation(text)
    text = replace_contractions(text)
    return text


def remove_html(sentence):
    return re.sub(re.compile('<.*?>'), ' ', sentence)


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans(dict.fromkeys(punctuation)))


def replace_specific_contractions(sentence):
    sentence = re.sub(r'won\'t', 'will not', sentence)
    sentence = re.sub(r'can\'t', 'can not', sentence)
    return sentence


def replace_general_contractions(sentence):
    sentence = re.sub(r'n\'t', ' not', sentence)
    sentence = re.sub(r'\'re', ' are', sentence)
    sentence = re.sub(r'\'s', ' is', sentence)
    sentence = re.sub(r'\'d', ' would', sentence)
    sentence = re.sub(r'\'ll', ' will', sentence)
    sentence = re.sub(r'\'t', ' not', sentence)
    sentence = re.sub(r'\'ve', ' have', sentence)
    sentence = re.sub(r'\'m', ' am', sentence)
    return sentence


def replace_contractions(sentence):
    sentence = replace_specific_contractions(sentence)
    sentence = replace_general_contractions(sentence)
    return sentence
