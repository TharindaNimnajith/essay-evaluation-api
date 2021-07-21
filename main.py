from fastapi import FastAPI

from controllers import essay_evaluation
from controllers import spelling_grammar_grammarbot

app = FastAPI()


@app.get('/')
async def root():
    return RedirectResponse(url='/docs')


@app.post('/essay')
def essay(essay: str):
    grammar = spelling_grammar_grammarbot.evaluate(essay)
    essay = essay_evaluation.evaluate()
    return {
        'grammar': grammar,
        'essay': essay
    }
