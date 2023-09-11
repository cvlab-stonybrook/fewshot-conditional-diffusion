# Segmentation Guidance Experiments

## FFHQ-256
We use the FFHQ-256 model from [Label-Efficient Semantic Segmentation with Diffusion Models](https://github.com/yandex-research/ddpm-segmentation) along with the few-shot data provided to generate segmentation-conditioned images.

``ffqh.ipynb`` showcases the full pipeline generate faces with segmentation guidance.

## Stable Diffusion
Generating images using Stable Diffusion guided with segmentation masks is demonsrtated in ``stable_diffusion.ipynb``. We again use the few-shot data from Baranchuk et al. to guide Stable Diffusion with segmentation masks. The notebook uses the Stable Diffusion v1.2 model from https://github.com/CompVis/latent-diffusion.
