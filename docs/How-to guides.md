## How to customize a new dataset?

None of my business, see hugging face [official document](https://huggingface.co/docs/datasets/v2.20.0/en/dataset_script#create-a-dataset-loading-script). Just place your dataset script into `testbed/data/<dataset_name>/<dataset_name>.py`, and implement `postprocess_generation` to clean and extract results from raw model output.

By the way, I prepared a convenient method in `testbed/data/common.py` to help you implement `_split_generator` of your new dataset.

## How to customize a new metric?

None of my business, see hugging face [official document](https://huggingface.co/docs/evaluate/creating_and_sharing). Just place your new metric into `testbed/evaluate/<metric_name>/<metric_name>.py`.

🚨 If you want replace official evaluation, you should add a test script in `test/` to prove that your code is consistent with official code.

## How to customize a new model?
You need to do follows:
1. Inherit from `ModelBase` that placed at `testbed/models/model_base.py`. It is just a simple wrapper for pretrained model and processor.
2. Implement `model_name` property to identify what the model is, and `default_prompt_template` which is used in `apply_prompt_tempalte` to transform raw texts and images to a model-specific prompt.
3. Implement `generate`. This method is almost same as in `transformers`, except applying prompt template, processing with `processor` and tokenizing.