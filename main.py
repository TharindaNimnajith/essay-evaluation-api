import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware import Middleware

from controllers import essay_evaluation
from controllers import spelling_grammar_grammarbot
from models.Essay import Essay

middleware = [
    Middleware(CORSMiddleware,
               allow_credentials=True,
               allow_origins=['*'],
               allow_methods=['*'],
               allow_headers=['*'])
]

app = FastAPI(middleware=middleware)


@app.get('/')
async def root():
    return RedirectResponse(url='/docs')


@app.post('/essay')
def essay(essay: Essay):
    spelling, grammar, matches = spelling_grammar_grammarbot.evaluate(essay.essay)
    essay_score = essay_evaluation.evaluate(essay.essay)
    score = (spelling * 25 + grammar * 25 + essay_score * 50) / 100
    return {
        'score': round(score),
        'essay_score': round(essay_score),
        'spelling': round(spelling),
        'grammar': round(grammar),
        'matches': matches
    }


if __name__ == '__main__':
    uvicorn.run(app,
                host='127.0.0.1',
                port=8001)
