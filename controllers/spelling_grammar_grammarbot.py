from grammarbot import GrammarBotClient

from .util import text_preprocessing

client = GrammarBotClient()


def evaluate(text):
    text = text_preprocessing.preprocess_text(text)
    try:
        res = client.check(text)
        matches = res.raw_json.get('matches')
        spelling_score, grammar_score = get_score(matches,
                                                  len(text.split()))
    except Exception as e:
        print(e)
        spelling_score, grammar_score, matches = 0, 0, []
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
