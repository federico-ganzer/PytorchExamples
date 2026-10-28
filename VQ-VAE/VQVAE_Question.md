Canopy Labs Coding ML Interview For ML Research

Attempt to complete the following exercise in 3 hours. You will take 45 minutes to understand the theory, and 15 minutes to discuss with a team member. You will have 105 minutes to code the solution, and 15 to discuss it with the team.

Overview

The VQ-VAE Paper (https://arxiv.org/pdf/1711.00937) illustrates how we can represent a continuous high dimensional manifold in a finite discrete representation space. 

This is a simplification of a modern and generally very powerful way for representing high dimensional multimodal data for generative tasks. 

We will take the MNIST set (https://huggingface.co/datasets/ylecun/mnist) and experiment using a VQVAE. 


Task:

- Using MSE as the loss train an autoencoder to reconstruct the data. What is the loss on an evaluation set?
- Train a VQVAE with 10 means to represent the data.
- Reconstruct each mean - what do you notice?
- Generally VQ-VAEs are quite lossy and do not reproduce realistic output data.
- What could it mean to describe a loss as smearing - what could this mean in the context of L^i norms for i = 0.5, 1, 2
- Adding a GAN can encourage the model to “hallucinate” realistic detail while roughly aligned with the reconstructive objective. Propose a GAN architecture and train a GAN to hallucinate realistic detail.
