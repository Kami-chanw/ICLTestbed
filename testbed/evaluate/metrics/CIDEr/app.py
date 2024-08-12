import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("Kamichanw/CIDEr")
launch_gradio_widget(module)