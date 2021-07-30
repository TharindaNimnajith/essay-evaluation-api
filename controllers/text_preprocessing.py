import re


def preprocess_text(text):
    text = str(text.strip())
    text = remove_html(text)
    return text


def remove_html(sentence):
    return re.sub(re.compile('<.*?>'), ' ', sentence)
