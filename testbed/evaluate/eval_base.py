import contextvars, contextlib
import json

ctx_eval = contextvars.ContextVar('ctx_eval')

class EvalBase(contextlib.ContextDecorator):
    def __init__(self, path):
        self.path = path
        self.records = []
        self.params = {}

    def __enter__(self):
        self.token = ctx_eval.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ctx_eval.reset(self.token)
        with open(self.path, 'w') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=4)

    def record(self, output, result):
        self.records.append({
            "output": output,
            "result": result
        })
    
    def add_paramters(self, **kwargs):
        self.params.update(**kwargs)