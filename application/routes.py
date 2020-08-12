from flask import current_app as app
from flask import request
from .model.generate import predict


@app.route('/')
def compendium():
    text = request.form.get('text')
    summary, _ = predict(text)
    summary = " ".join(summary[ii] for ii in range(len(summary[:-1])) if summary[ii - 1] != summary[ii])
    return {"summary": summary}
