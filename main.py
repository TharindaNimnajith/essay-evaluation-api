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
    grammar = spelling_grammar_grammarbot.evaluate(essay)
    essay = essay_evaluation.evaluate(essay)
    return {
        'grammar': grammar,
        'essay': essay
    }
