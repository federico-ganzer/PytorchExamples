Coding ML Interview For ML Research

Attempt to complete the following exercise in 3 hours. You will take 45 minutes to understand the theory, and 15 minutes to discuss with a team member. You will have 105 minutes to code the solution, and 15 to discuss it with the team.

## Overview

LFADS, Latent Factor Analysis via Dynamical Systems (Sussillo et al. 2016, https://arxiv.org/abs/1608.06315; Pandarinath et al. 2018, Nature Methods, https://doi.org/10.1038/s41592-018-0109-9), is a sequential VAE for spiking neural data. It assumes that on every single trial, the spikes of a recorded population are noisy Poisson observations of a low-dimensional latent dynamical system:

- A bidirectional RNN **encoder** reads the whole trial and outputs a posterior $q(g_0 \mid x) = \mathcal N(\mu, \mathrm{diag}\,\sigma^2)$ over the initial state of a generator.
- An autonomous **generator** RNN unrolls $g_0$ into states $g_t$, which are read out linearly into low-dimensional **factors** $f_t = W_f g_t$.
- Firing rates are $r_t = \exp(W_r f_t + b_r)$ and spikes are $s_t \sim \mathrm{Poisson}(r_t)$.
- The full model adds a **controller** RNN that infers time-varying inputs $u_t$ to the generator, for perturbations the initial condition can't explain.

Real recordings have no ground-truth latents. As in the paper, we use a synthetic population driven by a chaotic system, so we can check whether the inferred latents are right.

## Data

Generate it yourself (about 20 lines of numpy):

- Lorenz system with $\sigma = 10$, $\rho = 28$, $\beta = 8/3$. Integrate with RK4, $dt = 0.01$.
- 2000 trials. For each, draw $x_0 \sim \mathcal N((0, 0, 25), 10^2 I)$, burn in 1000 steps so the state is on the attractor, then record $T = 100$ time bins, one every 2 integration steps.
- z-score each of the 3 latent dimensions over the whole dataset, giving $z_t \in \mathbb R^3$.
- $N = 30$ neurons with $\log \lambda_{n,t} = \log(0.05) + c_n^\top z_t$, $c_n \sim \mathcal N(0, 0.5^2 I)$, and spikes $s_{n,t} \sim \mathrm{Poisson}(\lambda_{n,t})$. Read bins as 10 ms: the mean rate is about 8 Hz and about 93% of bins are empty.
- Split 1600 train / 400 test trials. Hold out the last 8 neurons. The model only ever sees the 22 held-in neurons as input, but has to predict all 30.

**Metric: co-smoothing bits per spike** (Neural Latents Benchmark, https://arxiv.org/abs/2109.04463). On test trials, infer latents from held-in neurons only and score the predicted rates $\hat r$ of the held-out neurons:

$$\text{co-bps} = \frac{\log p(s_{out} \mid \hat r) - \log p(s_{out} \mid \bar r)}{n_{spikes,out} \cdot \ln 2},$$

where $\bar r$ is each held-out neuron's mean rate on the training trials. Self-check: the true rates $\lambda$ score about 0.7 bits/spike, and Gaussian smoothing plus a Poisson GLM scores about 0.5. A small untuned LFADS (64-unit GRUs, 250 epochs) reaches about 0.58 and is still improving, so there is room to tune.

## Task

- Baseline: smooth the held-in spikes with a Gaussian kernel and fit a Poisson GLM from the smoothed held-in activity to each held-out neuron. What is the test co-bps? Sweep the kernel width. What trade-off are you seeing?
- Now score smoothed spikes against the *same* neurons they were computed from. Why does this beat the true rates, and why is the number meaningless?
- Implement LFADS without the controller: bidirectional GRU encoder → $q(g_0 \mid x)$ → generator GRU with no input → 8 factors → log-rates for all 30 neurons. Train on the ELBO (Poisson log-likelihood minus KL to $\mathcal N(0, I)$) with KL warm-up. Report test co-bps.
- Fit a linear map, on training trials, from the inferred factors to the true Lorenz $z_t$. What is the test $R^2$ per dimension? Plot inferred against true trajectories for a few single trials, and plot the 3-D attractor. What do you notice? Why can we only hope to recover $z$ up to an affine transformation, or worse?
- Plot single-trial rates for one held-out neuron: true $\lambda$, the smoothed baseline, and LFADS. Where does LFADS beat smoothing, and why?
- Sample $g_0 \sim \mathcal N(0, I)$ from the prior and run the generator for 300 steps, three times the training length. Does it look like the Lorenz attractor? What happens after $T = 100$?
- Break it: train with 1, 2, 3 and 8 factors and plot co-bps against factor dimension. Train without KL warm-up and without dropout. Shrink the generator. What fails, and how does it show up?

## Mathematical insight

- Write the LFADS ELBO and derive the closed-form KL between $\mathcal N(\mu, \mathrm{diag}\,\sigma^2)$ and $\mathcal N(0, I)$. Why does the reparameterisation trick matter here?
- Given $g_0$, the generator is deterministic, so the only per-trial randomness is $g_0$. The Lorenz system's largest Lyapunov exponent is about 0.9. How much does an error in $g_0$ grow over one trial (100 bins = 2 time units)? What does that predict for inference late in a trial, and for longer trials? How do the controller's inferred inputs $u_t$ change the picture?
- Poisson vs MSE: write $-\log p(s \mid \lambda)$ for a Poisson. Show that MSE on raw counts is the NLL of a Gaussian with fixed variance, and explain why that mis-weights neurons and time bins when variance equals mean. Show that $\sqrt{s}$ (Anscombe: $2\sqrt{s + 3/8}$) approximately stabilises the variance. Compare with the "smearing" question in the VQ-VAE exercise.
- A flexible enough sequential autoencoder can drive the training NLL down by copying each spike from input to output. Explain why this is fitting noise, and why the co-smoothing metric can't be gamed this way. Coordinated dropout (Keshtkaran & Pandarinath 2019, https://arxiv.org/abs/1908.07896) drops a random subset of input entries and computes the loss only on those dropped entries. Show that this makes the identity solution useless.

## Stretch

- Add unobserved input pulses: on each trial, at one random time $t^* \in [20, 80]$, add a kick $\delta \sim \mathcal N(0, 10^2 I)$ to the raw Lorenz state. Show that LFADS without a controller can't fit the kicks. Add the controller RNN with a Gaussian (or autoregressive) prior on $u_t$. Does the inferred $u_t$ peak at $t^*$?
- Real data: MC_Maze from the Neural Latents Benchmark (DANDI 000128). How does your model compare with the published co-bps numbers?
