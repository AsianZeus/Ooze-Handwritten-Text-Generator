# Ooze - Handwritten Text Generator
This is an TensorFlow Implementation of Ooze - Handwritten Text Generator by Akshat Surolia, This is an Image to Image translation approach to generate realistic hand written sentences by taking text as an input.

## Training 
    - The model is trained on the custom dataset created by authors of the paper.
    - Model is trained on 2 V100 GPUs with standard Cycle GAN Architecture.
    - API is made with Flask and Jinja2 templates, available for training and inference.

---
## Abstract
>In this paper we show how Generative Adversarial Network (Goodfellow et al., 2014), more specifically Cycle-GAN(Zhu et al., 2017), can be used for Human like handwriting generation with output size up to a text line. The methodology used in Cycle-GAN is to establish translation between two different domains (e.g., connection between image of horse and zebra), extending this methodology to establishment of translation between machine printed text and handwritten text is mentioned in this paper. The neural network used in this paper is trained for dataset created by the author and can be trained with another dataset.

---
## How to Cite

    Surolia, A. (2021) “Ooze - Handwritten Text Generator”,
    GLS KALP – Journal of Multidisciplinary Studies, 1(4), pp. 35–49.
    Available at: http://glskalp.in/index.php/glskalp/article/view/19.
