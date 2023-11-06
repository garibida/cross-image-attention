# Cross-Image Attention for Zero-Shot Appearance Transfer

> Yuval Alaluf*, Daniel Garibi*, Or Patashnik, Hadar Averbuch-Elor, Daniel Cohen-Or  
> Tel Aviv University  
> \* Denotes equal contribution  
>
> Recent advancements in text-to-image generative models have demonstrated a remarkable ability to capture a deep semantic understanding of images. In this work, we leverage this semantic knowledge to transfer the visual appearance between objects that share similar semantics but may differ significantly in shape. To achieve this, we build upon the self-attention layers of these generative models and introduce a cross-image attention mechanism that implicitly establishes semantic correspondences across images. Specifically, given a pair of images ––– one depicting the target structure and the other specifying the desired appearance ––– our cross-image attention combines the queries corresponding to the structure image with the keys and values of the appearance image. This operation, when applied during the denoising process, leverages the established semantic correspondences to generate an image combining the desired structure and appearance. In addition, to improve the output image quality, we harness three mechanisms that either manipulate the noisy latent codes or the model's internal representations throughout the denoising process. Importantly, our approach is zero-shot, requiring no optimization or training. Experiments show that our method is effective across a wide range of object categories and is robust to variations in shape, size, and viewpoint between the two input images.

<!-- <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv.svg" height=22.5></a> -->
<a href="https://garibida.github.io/cross-image-attention/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=20.5></a>

<p align="center">
<img src="docs/teaser.jpg" width="90%"/>  
<br>
Given two images depicting a source structure and a target appearance, our method generates an image merging the structure of one image with the appearance of the other in a zero-shot manner.
</p>

# Code Coming Soon!

## Appearance Transfer Results 
<p align="center">
<img src="docs/grids.jpg" width="90%"/>  
<br>
<img src="docs/general_results.jpg" width="90%"/>  
<br>
Sample appearance transfer obtained by our cross-image attention technique.
</p>

<p align="center">
<img src="docs/cross_domain.jpg" width="90%"/>  
<br>
Our method can also be used to transfer appearance between cross-domain objects.
</p>


## Citation
If you use this code for your research, please cite the following work: 
```
@misc{
}
```
