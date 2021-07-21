import re
from string import punctuation

from grammarbot import GrammarBotClient

client = GrammarBotClient()


def evaluate(text):
    text = preprocess_text(text)
    res = client.check(text)
    matches = res.raw_json.get('matches')
    spelling_score, grammar_score = get_score(matches, len(text.split()))
    return spelling_score, grammar_score, matches


def get_score(matches, length):
    spelling_mistakes_count = 0
    grammar_mistakes_count = 0
    for match in matches:
        if match['rule']['issueType'] == 'misspelling':
            spelling_mistakes_count += 1
        else:
            grammar_mistakes_count += 1
    if spelling_mistakes_count == 0:
        spelling_score = 10.0
    else:
        spelling_score = (length - spelling_mistakes_count) * 7.5 / length
    if grammar_mistakes_count == 0:
        grammar_score = 10.0
    else:
        grammar_score = (length - grammar_mistakes_count) * 7.5 / length
    return spelling_score, grammar_score


def preprocess_text(text):
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
