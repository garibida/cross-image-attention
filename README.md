# Cross-Image Attention for Zero-Shot Appearance Transfer

> Yuval Alaluf*, Daniel Garibi*, Or Patashnik, Hadar Averbuch-Elor, Daniel Cohen-Or  
> Tel Aviv University  
> \* Denotes equal contribution  
>
> Recent advancements in text-to-image generative models have demonstrated a remarkable ability to capture a deep semantic understanding of images. In this work, we leverage this semantic knowledge to transfer the visual appearance between objects that share similar semantics but may differ significantly in shape. To achieve this, we build upon the self-attention layers of these generative models and introduce a cross-image attention mechanism that implicitly establishes semantic correspondences across images. Specifically, given a pair of images ––– one depicting the target structure and the other specifying the desired appearance ––– our cross-image attention combines the queries corresponding to the structure image with the keys and values of the appearance image. This operation, when applied during the denoising process, leverages the established semantic correspondences to generate an image combining the desired structure and appearance. In addition, to improve the output image quality, we harness three mechanisms that either manipulate the noisy latent codes or the model's internal representations throughout the denoising process. Importantly, our approach is zero-shot, requiring no optimization or training. Experiments show that our method is effective across a wide range of object categories and is robust to variations in shape, size, and viewpoint between the two input images.

<a href="https://arxiv.org/abs/2311.03335"><img src="https://img.shields.io/badge/arXiv-2311.03335-b31b1b.svg" height=22.5></a>
<a href="https://garibida.github.io/cross-image-attention/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=20.5></a>
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yuvalalaluf/cross-image-attention)

<p align="center">
<img src="docs/teaser.jpg" width="90%"/>  
<br>
Given two images depicting a source structure and a target appearance, our method generates an image merging the structure of one image with the appearance of the other in a zero-shot manner.
</p>


## Description  
Official implementation of our Cross-Image Attention and Appearance Transfer paper.


## Environment
Our code builds on the requirement of the `diffusers` library. To set up their environment, please run:
```
conda env create -f environment/environment.yaml
conda activate cross_image
```

## Usage  
<p align="center">
<img src="docs/general_results.jpg" width="90%"/>  
<br>
Sample appearance transfer results obtained by our cross-image attention technique.
</p>

To generate an image, you can simply run the `run.py` script. For example,
```
python run.py \
--app_image_path /path/to/appearance/image.png \
--struct_image_path /path/to/structure/image.png \
--output_path /path/to/output/images.png \
--domain_name [domain the objects are taken from (e.g., animal, building)] \
--use_masked_adain True \
--contrast_strength 1.67 \
--swap_guidance_scale 3.5 \
```
Notes:
- To perform the inversion, if no prompt is specified explicitly, we will use the prompt `"A photo of a [domain_name]"`
- If `--use_masked_adain` is set to `True` (its default value), then `--domain_name` must be given in order 
  to compute the masks using the self-segmentation technique.
  - In cases where the domains are not well-defined, you can also set `--use_masked_adain` to `False` and 
    no `domain_name` is required.
- You can set `--load_latents` to `True` to load the latents from a file instead of inverting the input images every time. 
  - This is useful if you want to generate multiple images with the same structure but different appearances.


### Demo Notebook 
<p align="center">
<img src="docs/grids.jpg" width="90%"/>  
<br>
Additional appearance transfer results obtained by our cross-image attention technique.
</p>

We also provide a notebook to run in Google Colab, please see `notebooks/demo.ipynb`.


## HuggingFaceDemo :hugs:
We also provide a simple HuggingFace demo to run our method on your own images.   
Check it out [here](https://huggingface.co/spaces/yuvalalaluf/cross-image-attention)!


## Acknowledgements 
This code builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library. In addition, we 
borrow code from the following repositories: 
- [Edit-Friendly DDPM Inversion](https://github.com/inbarhub/DDPM_inversion) for inverting the input images.
- [Prompt Mixing](https://github.com/orpatashnik/local-prompt-mixing) for computing the masks used in our AdaIN operation.
- [FreeU](https://github.com/ChenyangSi/FreeU) for improving the general generation quality of Stable Diffusion.


## Citation
If you use this code for your research, please cite the following work: 
```
@misc{alaluf2023crossimage,
      title={Cross-Image Attention for Zero-Shot Appearance Transfer}, 
      author={Yuval Alaluf and Daniel Garibi and Or Patashnik and Hadar Averbuch-Elor and Daniel Cohen-Or},
      year={2023},
      eprint={2311.03335},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
