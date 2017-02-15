---
title: "Perplexity and Intrinsic Dimensionality"
author: "James Melville"
date: "February 7, 2017"
output: html_document
---

Lee and co-workers' work on
[multi-scale neighbor embedding](https://dx.doi.org/10.1016/j.neucom.2014.12.095)
makes use of the concept of intrinsic dimensionality, which relates the 
perplexity of the input probability and the bandwidth of the gaussian similarity
function. They provide an expression based on one-sided finite differences,
which is convenient for multi-scaled neighbor embedding because you calculate
the input probabilities for multiple perplexities.

First, they show that, assuming the data is uniformly distributed, the relation 
between the precision parameter of a Gaussian similarity function, $\beta$, the 
Shannon Entropy of the resulting probability distribution, $H$, and the
intrinsic dimensionality, $D$, is:

$$
H = -\frac{D}{2} \log_2 \left( \beta \right) + C
$$

where $C$ is some constant that we don't care about. That is, if you plotted 
Shannon Entropy against $\log_2\left(\beta\right)$ you should get a straight 
line, with the gradient being $-D/2$.

The finite difference expression for the dimensionality of point $i$, $D_i$ is:

$$
D_i = \frac{2}{\log_2\left(\beta_{U,i}\right)-\log_2\left(\beta_{V,i}\right)}
$$

where $\beta_{U,i}$ is the precision parameter for the gaussian function which
generates the $i$th similarity/weight for some perplexity, $U$:

$$w_{U,ij} = \exp\left(-\beta_{\left(U,i\right)}d^2_{ij}\right)$$

$d^2_{ij}$ is the squared distance between point $i$ and $j$ in the input
distances. 

### Speculation Alert

The following is something I just came up with myself. It's *not* in the paper 
by Lee and co-workers, so take it for what it's worth (probably not much). 

If you consider an analytical expression for 
$\partial H/\partial \log \left(\beta_i\right)$, using similar chain rule 
expressions and nomenclature that I went into in great and tedious detail in the 
[SNE gradient](gradients.html) section, you can get to an expression which only 
requires one perplexity value:

$$
D_i = -2 \beta_i \sum_j d^2_{ij} p_{j|i} \left[\log\left(p_{j|i}\right) + H\right]
$$

$p_{j|i}$ is the conditional input probability of picking point $j$ as a 
neighbor of $i$, and $H$ is the corresponding Shannon Entropy (in nats) of the 
corresponding probability distribution, which is related to the perplexity
by $U = \log(H)$.

This could be useful for estimating the intrinsic dimensionality associated
with a perplexity even when not using multiscaling.
