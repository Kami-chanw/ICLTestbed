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
I have divided the in-context learning process into four stages: data loading, model setup, model inference, and evaluation. These stages correspond to the three main modules of ICLTestbed: `testbed.data`, `testbed.models`, and `testbed.evaluate`. If you are interested in visual question answering task, you can see the complete usage process in [this tutorial](./examples/tutorial_vqa.ipynb). For image captioning task, see [this tutorial](./examples/tutorial_caption.ipynb)

If you want to customize new datasets, models, or metrics, you can follow the suggestions in the [How-to guides](./docs/How-to%20guides.md), or directly raise an issue for me to implement it.

## Overview of Supported Components

<table style="width: 100%; border-collapse: collapse; text-align: center;">
  <thead>
    <tr style="border-bottom: 2px solid white; font-weight: bold; vertical-align: middle;">
      <th style="text-align: center;">Task</th>
      <th style="text-align: center;">Dataset</th>
      <th style="text-align: center;">Model</th>
      <th style="text-align: center;">Metrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="vertical-align: middle;">Visual Question Answering</td>
      <td>VQA v2</td>
      <td rowspan="3" style="vertical-align: middle;">Idefics <br> Idefics2</td>
      <td rowspan="2">vqa accuracy</td>
    </tr>
    <tr>
      <td>OK-VQA</td>
    </tr>
    <tr>
      <td style="vertical-align: middle;">Image Captioning</td>
      <td>COCO (Karpathy split)</td>
      <td>CIDEr</td>
    </tr>
  </tbody>
</table>



## License
This project uses the MIT License.