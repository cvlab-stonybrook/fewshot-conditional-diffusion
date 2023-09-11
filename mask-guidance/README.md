# Segmentation Guidance Experiments

## FFHQ-256
We use the FFHQ-256 model from [Label-Efficient Semantic Segmentation with Diffusion Models](https://github.com/yandex-research/ddpm-segmentation) along with the few-shot data provided to generate segmentation-conditioned images.

``ffhq.ipynb`` showcases the full pipeline generate faces with segmentation guidance.

## Stable Diffusion
Generating images using Stable Diffusion guided with segmentation masks is implemented in ``stable_diffusion.ipynb``. We again use the few-shot data (LSUN-Cat) from Baranchuk et al. to guide Stable Diffusion with segmentation masks. The notebook uses the Stable Diffusion v1.2 model. To run you need to first install the LDM environment, as described in the official Stable Diffusion [repository](https://github.com/CompVis/latent-diffusion).
