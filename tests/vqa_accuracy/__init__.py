from .vqa import VQA
from .vqaEval import VQAEval
import random

def generate_fake_result(questions_path, annotations_path):
    vqa = VQA(annotations_path, questions_path)
    
