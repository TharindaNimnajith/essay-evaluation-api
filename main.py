from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from controllers import essay_evaluation
from controllers import spelling_grammar_grammarbot

app = FastAPI()


@app.get('/')
async def root():
    return RedirectResponse(url='/docs')


@app.post('/essay')
def essay(essay: str):
    spelling, grammar, matches = spelling_grammar_grammarbot.evaluate(essay)
    essay_score = essay_evaluation.evaluate(essay)
    score = int((spelling + grammar + essay_score) / 3)
    return {
        'spelling': spelling,
        'grammar': grammar,
        'matches': matches,
        'essay_score': essay_score,
        'score': score
    }
