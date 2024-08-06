# ICLTestbed: In-context Learning Testbed Designed for Researchers

## Introduction
ICLTestbed is a framework specifically designed for researchers working on in-context learning. It encapsulates various commonly used models and datasets in a clear and concise process, aiming to free researchers from complex engineering code and allowing them to focus on creative research.

<details open>
<summary>Major features</summary>

- **Modular design**: 

  The in-context learning framework is decomposed into different components, enabling users to easily construct a customized in-context learning framework by combining different modules.

- **Low cost of getting started**:

  Most components are directly based on Hugging Face's library or PyTorch, allowing users to get started with very low learning costs.

- **Support for multiple models and tools out of the box**:

  The toolbox directly supports multiple models, datasets, and metrics.


</details>

## Getting Started
I have divided the in-context learning process into four stages: data loading, model setup, model inference, and evaluation. These stages correspond to the three main modules of ICLTestbed: `testbed.data`, `testbed.models`, and `testbed.evaluate`. You can see the complete usage process in [this tutorial](./examples/tutorial.ipynb).

If you want to customize new datasets, models, or metrics, you can follow the suggestions in the [How-to guides](./docs/How-to%20guides.md), or directly raise an issue for me to implement it.