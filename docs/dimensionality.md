---
title: "Perplexity and Intrinsic Dimensionality"
date: "February 7, 2017"
output:
  html_document:
    theme: cosmo
---

Up: [Index](index.html)

## Soft Correlation Dimension

Choosing the correct perplexity value for an embedding is an open research
problem. The usual approach is to plump for a value around 30. The
[How to Use t-SNE Effectively](http://distill.pub/2016/misread-tsne/) web site
demonstrates that multiple perplexity values can give quite different results
even for simple synthetic datasets.

Lee and co-workers' work on
[multi-scale neighbor embedding](https://dx.doi.org/10.1016/j.neucom.2014.12.095)
attempts to get around this problem by considering multiple perplexities during
optimization. Part of their discussion involves making use of the concept of
intrinsic dimensionality of the input data and relating that to perplexity,
based on the concept of
[correlation dimension](https://doi.org/10.1007/BF01058436).

For estimating the intrinsic dimensionality of the input data, they provide an
expression based on one-sided finite differences, which is convenient for
multi-scaled neighbor embedding because you calculate the input probabilities
for multiple perplexities.

Here I'm going to derive an analytical expression for the dimensionality with
the use of the exponential kernel (exponential with regard to the squared
distances, or gaussian in the raw distances, if you prefer) that only requires
one perplexity calculation.

First, some definitions. This discussion of dimensionality involves the input
probabilities, so we will assume the use of gaussian similarities, not anything
exotic like the t-distribution or other heavy-tailed functions:

$$
w_{ij} = \exp(-\beta_i d_{ij}^{2})
$$

where $d_{ij}^{2}$ is the squared distance between point $i$ and $j$ in the
input space and $\beta_i$ is the precision (inverse of the bandwidth)
associated with point $i$. We use a point-wise normalization to define a
probability, $p_{j|i}$:

$$
p_{j|i} = \frac{w_{ij}}{\sum_{k} w_{ik}}
$$

The Shannon Entropy, $H$, of the probability distribution is:

$$
H = -\sum_{j} p_{j|i} \log p_{j|i}
$$

and the perplexity, $U$, in units of nats is:

$$
U = \exp(H)
$$

Lee and co-workers show that, assuming the data is uniformly distributed, the
relationship between the precision, perplexity and intrinsic dimensionality
around point $i$, $D_i$, is:

$$
U \propto \beta_i^{-\left({D_i}/{2}\right)}
$$

*10 November 2022* Here are some extra steps to make that more obvious. First,
consider the Gaussian using the bandwidth, $\sigma$:

$$w_{ij} = \exp(-\frac{1}{\sigma^2} d_{ij}^{2})$$

i.e.:

$$\beta_i = 1 / \sigma_i^2 \implies \sigma_i = 1 / \sqrt{\beta_i} = \beta_i^{-1/2} $$

Often you see a factor of two in there (as indeed in the Lee paper) but that's
not important here.

The bandwidth, $\sigma_i$ is suggested to be analogous to the radius $R$ of a
hard ball. Similarly, the perplexity, due to its connection to number of nearest
neighbors, is analogous to the volume. So in the same way that the
[volume of an n-ball](https://en.wikipedia.org/wiki/Volume_of_an_n-ball) is
proportional to $R^n$, Lee and co-workers suggest that perplexity should be
proportional to the bandwidth raised to the power of the intrinsic
dimensionality:

$$
U \propto \sigma_i^{D_i} \\
U = c\sigma_i^{D_i} \\
\log U = D_i \log \sigma_i + \log c \\
\log U = D_i \log \left(\beta_i^{-1/2} \right) + \log c \\
\log U = -\frac{D_i}{2} \log \beta_i + \log c
$$

This suggests that if you did a log-log plot of the perplexity against the
precision for multiple perplexities, the graph should be linear and the gradient
would be $-D_{i}/2$ (with $\log c$ as the intercept, which is related to the
local density).

With a multi-scaling approach, a finite difference expression using base 2
logarithms is given as:

$$
D_i = \frac{2}{\log_2\left(\beta_{U,i}\right)-\log_2\left(\beta_{V,i}\right)}
$$

where $\beta_{U,i}$ is the precision parameter for the gaussian function which
generates the $i$th similarity/weight for some perplexity, $U$, where $U$ is an
exact power of 2 (e.g. $U = 16$) and $V$ is the perplexity for the next power
of 2 (e.g. $V = 32$), which leads to a convenient cancellation in the numerator.
This is handy in the case of the multi-scaling scheme give by Lee and co-workers
as successive powers of two of the perplexity are already being used in the
embedding. But a generic one-sided finite difference:

$$
D_i = -2\frac{\log U - \log V}{\log\left(\beta_{U,i}\right)-\log\left(\beta_{V,i}\right)}
$$

with any base logarithm and where $U$ and $V$ can be arbitrarily chosen
(presumably you would want them quite close to minimize error), should work just
as well.

## An Analytical Expression for Intrinsic Dimensionality

The following is something I just came up with myself. It's *not* in the paper
by Lee and co-workers, so take it for what it's worth (probably not much).

The gradient of the log-log plot of the perplexity against the precision is:

$$
\frac{\partial \log U}{\partial \log \beta_i} =
\beta_i \frac{\partial H}{\partial \beta_i} =
-\frac{D_{i}}{2}
$$

so:

$$
D_i = -2 \beta_i \frac{\partial H}{\partial \beta_i}
$$

We therefore need to find an expression for $\partial H / \partial \beta_{i}$.
Fortunately, we can use similar chain rule expressions and nomenclature that I
went into in great and tedious detail in the discussion of deriving the
[SNE gradient](gradients.html).

To start, we can write the derivative as:

$$
 \frac{\partial H}{\partial \beta_i} =
 \sum_{jklm}
 \frac{\partial H}{\partial p_{k|j}}
 \frac{\partial p_{k|j}}{\partial w_{lm}}
 \frac{\partial w_{lm}}{\partial \beta_{i}}
$$

$\partial w_{lm} / \partial \beta_{i} = 0$ unless $l = i$. Additionally, due to
the point-wise normalization, $\partial p_{k|j} / \partial w_{lm} = 0$ unless
$j = l$, allowing us to simplify the summation to:

$$
 \frac{\partial H}{\partial \beta_i} =
 \sum_{km}
 \frac{\partial H}{\partial p_{k|i}}
 \frac{\partial p_{k|i}}{\partial w_{im}}
 \frac{\partial w_{im}}{\partial \beta_{i}}
$$
Now let us regroup that double summation into two single summations, and also
rename $m$ to $j$:

$$
 \frac{\partial H}{\partial \beta_i} =
 \sum_{j}
 \left[
   \sum_{k}
   \frac{\partial H}{\partial p_{k|i}}
   \frac{\partial p_{k|i}}{\partial w_{ij}}
 \right]
 \frac{\partial w_{ij}}{\partial \beta_{i}}
$$
We are now in very familiar territory if you have read the
[SNE gradient](gradients.html) page. The full details of how to derive the
expression of the gradient of the weight normalization is on that page, but we
shall jump straight to inserting the result for
$\partial p_{k|i} / \partial w_{ij}$
to get:

$$
 \frac{\partial H}{\partial \beta_i} =
 \sum_{j}
 \frac{1}{S_{i}}
 \left[
   \frac{\partial H}{\partial p_{j|i}}
   -\sum_{k}
   \frac{\partial H}{\partial p_{k|i}}
   p_{k|i}
 \right]
 \frac{\partial w_{ij}}{\partial \beta_{i}}
$$
where $S_{i}$ is:

$$
S_{i} = \sum_{j} w_{ij}
$$

The gradient of the Shannon Entropy with respect to the probability is:

$$
\frac{\partial H}{\partial p_{j|i}} =
- \log \left( p_{j|i} \right) - 1
$$
substituting into the expression in square brackets:

$$
 \left[
   \frac{\partial H}{\partial p_{j|i}}
   -\sum_{k}
   \frac{\partial H}{\partial p_{k|i}}
   p_{k|i}
 \right]
 =
 \left[
- \log \left( p_{j|i} \right) - 1
   -\sum_{k}
   \left\{
   - \log \left( p_{k|i} \right) - 1
   \right\}
   p_{k|i}
 \right]
  =
 \left[
   - \log \left( p_{j|i} \right) - 1
   +\sum_{k}
   p_{k|i} \log \left( p_{k|i} \right)
   +\sum_{k}
   p_{k|i}
 \right]
$$

and because $\sum_k p_{k|i} = 1$, we eventually get to:

$$
 \left[
   \frac{\partial H}{\partial p_{j|i}}
   -\sum_{k}
   \frac{\partial H}{\partial p_{k|i}}
   p_{k|i}
 \right]
 =
 \left[
- \log \left( p_{j|i} \right) - H
 \right]
 =
 -\left[
  \log \left( p_{j|i} \right) + H
 \right]
$$

At this point:

$$
 \frac{\partial H}{\partial \beta_i} =
 \sum_{j}
 -\frac{1}{S_{i}}
 \frac{\partial w_{ij}}{\partial \beta_{i}}
  \left[
  \log \left( p_{j|i} \right) + H
 \right]
$$

which leads to:

$$
D_{i} = \frac{2 \beta_i}{S_i}
 \sum_{j}
 \frac{\partial w_{ij}}{\partial \beta_{i}}
  \left[
  \log \left( p_{j|i} \right) + H
 \right]
$$

The gradient of the exponential weight with respect to the precision parameter, $\beta_i$,
is:

$$
\frac{\partial w_{ij}}{\partial \beta_{i}}
=
-d_{ij}^2 w_{ij}
$$

Substituting these two sub expressions into the total gradient, we are left with:
$$
 \frac{\partial H}{\partial \beta_i} =
 \sum_{j}
 \frac{d_{ij}^2 w_{ij}}{S_{i}}
 \left[
 \log \left( p_{j|i} \right) + H
 \right]
$$

Additionally, $p_{j|i} = w_{ij} / S_{i}$, so the final expression for the
gradient is:

$$
 \frac{\partial H}{\partial \beta_i} =
 \sum_{j}
 d_{ij}^2 p_{j|i}
 \left[
 \log \left( p_{j|i} \right) + H
 \right]
$$

The only thing to left to do is to multiply this expression by $-2 \beta_{i}$
to get this expression for the intrinsic dimensionality:

$$
D_i = -2 \beta_i \sum_j d^2_{ij} p_{j|i} \left[\log\left(p_{j|i}\right) + H\right]
$$

which is useful if you've carried out the perplexity-based calibration
on the input weights, as you have already calculated $p_{j|i}$, $H$ and
$\beta_i$.

## Un-normalized weights

If you'd rather think in terms of the un-normalized weights and the distances
only, the usual expression for Shannon entropy can be rewritten in terms of
weights as:

$$
H = \log S_{i} -\frac{1}{S_i} \left( \sum_j w_{ij} \log w_{ij} \right)
$$

and that can be combined with:

$$
\log \left( p_{j|i} \right) = \log \left( w_{ij} \right) + \log \left(S_i \right)
$$

to give:

$$
D_{i} = \frac{2 \beta_i}{S_i^2}
 \sum_{j}
 \frac{\partial w_{ij}}{\partial \beta_{i}}
  \left(
   S_i \log w_{ij} - \sum_k w_{ik} \log w_{ik}
 \right)
$$

We can also express the relation between the weight and the squared distance as:

$$
\log \left( w_{ij} \right) = -\beta_i d_{ij}^2
$$
and the gradient with respect to the precision as:

$$
\frac{\partial w_{ij}}{\partial \beta_{i}}
=
-d_{ij}^2 w_{ij} = -\frac{w_{ij} \log w_{ij}}{\beta_i}
$$

and the Shannon entropy expression as:

$$
H = \log S_i + \frac{\beta_i}{S_i} \sum_j d_{ij}^2 w_{ij}
$$

With all that, you can eventually get to two equivalent expressions for $D_i$:

$$
D_{i} = \frac{2}{S_i}
\left\{
\sum_j w_{ij} \left[ \log \left( w_{ij} \right) \right] ^ 2
-\frac{1}{S_i} \left[ \sum_j w_{ij} \log \left( w_{ij} \right) \right]^2
\right\}
$$
$$
D_{i} = \frac{2 \beta_i^2}{S_i}
\left[
\sum_j d_{ij}^4 w_{ij}
-\frac{1}{S_i} \left( \sum_j d_{ij}^2 w_{ij} \right)^2
\right]
$$

The first one only requires the $W$ matrix, although as you've been carrying out
lots of exponential operations to generate the weights, it seems a pity to have
to then carry out lots of expensive log calculations, in which case the second
expression might be better, but which requires the squared distance matrix and
$\beta_i$ also.

## Testing with Gaussians

*10 November 2022*. Below are some values of the intrinsic dimensionality
estimated at a series of perplexities, using 100,000 points sampled from
Gaussians of increasing dimensionality. I used the `gaussian_data` function in 
the [snedata](https://github.com/jlmelville/snedata) package to generate the 
Gaussians with `n = 100000, sdev = 1` for `dim` from 1 to 10 and also for 
`dim = 50`.

Results below are calculated using the values for $p_{j|i}$ and $\beta_i$ from
C++ code in [uwot](https://github.com/jlmelville/uwot) with the 150 exact 
nearest neighbor distances calculated with the `brute_force_knn` function in 
[rnndescent](https://github.com/jlmelville/uwot). $D_i$ was calculated for each
point, and the mean average across the whole dataset is reported below. The
maximum value of $D_i$ attained for each dataset, recommended as the estimator
for the intrinsic dimensionality by Lee and co-workers, is shown in bold.

| U     | 1        | 2        | 3        | 4        | 5        | 6        | 7        | 8        | 9        | 10       | 50       |
|-------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 5     | **1.16** | 1.98     | 2.63     | 3.15     | 3.57     | 3.90     | 4.16     | 4.38     | 4.54     | 4.69     | 5.63     |
| 10    | 1.09     | **2.04** | 2.88     | 3.61     | 4.21     | 4.68     | 5.04     | 5.33     | 5.55     | 5.74     | 6.89     |
| 15    | 1.07     | 2.04     | **2.95** | 3.75     | 4.41     | 4.91     | **5.28** | **5.57** | **5.79** | **5.98** | **7.11** |
| 20    | 1.05     | 2.03     | 2.97     | **3.79** | **4.43** | **4.92** | 5.27     | 5.55     | 5.76     | 5.93     | 6.99     |
| 25    | 1.04     | 2.03     | 2.98     | 3.77     | 4.37     | 4.82     | 5.15     | 5.41     | 5.60     | 5.76     | 6.72     |
| 30    | 1.03     | 2.02     | 2.96     | 3.70     | 4.26     | 4.67     | 4.97     | 5.20     | 5.37     | 5.52     | 6.40     |
| 35    | 1.03     | 2.02     | 2.92     | 3.61     | 4.12     | 4.49     | 4.75     | 4.96     | 5.12     | 5.25     | 6.05     |
| 40    | 1.03     | 2.01     | 2.86     | 3.49     | 3.95     | 4.28     | 4.52     | 4.71     | 4.85     | 4.97     | 5.69     |
| 45    | 1.03     | 2.00     | 2.79     | 3.36     | 3.77     | 4.06     | 4.28     | 4.45     | 4.57     | 4.68     | 5.34     |
| 50    | 1.02     | 1.97     | 2.70     | 3.21     | 3.57     | 3.84     | 4.03     | 4.18     | 4.30     | 4.39     | 4.98     |

To check if the analytical expression is correct, for comparison, here are the
one-sided finite difference results (so, no results for perplexity 50). These
don't use the increasing-powers-of-two expression from the Lee paper, just the
next largest perplexity as shown in the table, e.g. the result for $U=5$ uses
the $\beta_i$ values from $U = 5$ and $U = 10$.

| U  | 1        | 2        | 3        | 4        | 5        | 6        | 7        | 8        | 9        | 10       | 50       |
|:--:|:--------:| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 5  | **1.09** | 2.00     | 2.75     | 3.38     | 3.90     | 4.30     | 4.62     | 4.87     | 5.07     | 5.24     | 6.31     |
| 10 | 1.07     | **2.03** | 2.92     | 3.69     | 4.32     | 4.81     | 5.18     | 5.47     | 5.69     | 5.88     | 7.04     |
| 15 | 1.06     | 2.03     | 2.96     | 3.78     | **4.43** | **4.92** | **5.29** | **5.57** | **5.79** | **5.97** | **7.07** |
| 20 | 1.04     | 2.03     | **2.98** | **3.79** | 4.41     | 4.88     | 5.22     | 5.49     | 5.68     | 5.85     | 6.87     |
| 25 | 1.04     | 2.03     | 2.97     | 3.74     | 4.32     | 4.75     | 5.07     | 5.31     | 5.49     | 5.64     | 6.57     |
| 30 | 1.03     | 2.02     | 2.94     | 3.66     | 4.19     | 4.58     | 4.87     | 5.09     | 5.25     | 5.39     | 6.23     |
| 35 | 1.03     | 2.02     | 2.89     | 3.55     | 4.03     | 4.39     | 4.64     | 4.84     | 4.99     | 5.11     | 5.88     |
| 40 | 1.03     | 2.00     | 2.83     | 3.43     | 3.86     | 4.17     | 4.40     | 4.58     | 4.71     | 4.83     | 5.52     |
| 45 | 1.02     | 1.99     | 2.75     | 3.29     | 3.67     | 3.95     | 4.16     | 4.32     | 4.44     | 4.54     | 5.16     |

Results are very similar to the analytical values so probably the analytical
expression is correct.

From the table, the intrinsic dimensionality estimate seems accurate up to
around $D = 3$ and $D = 4$, after which the estimate progressively under-states
the true dimensionality. This is probably due to known edge effects which occur
with these sorts of estimate of intrinsic dimensionality and which become
increasingly troublesome at higher dimensions where proportionally more and more
of the data concentrates at the edges, requiring exponentially more data. In
the case of the 10D Gaussian, dropping the number of points by an order of
magnitude to 10,000 points reduces the intrinsic dimensionality estimate to
5.80. Increasing by an order of magnitude to 1,000,000 points increases the
estimate to 6.04. Clearly, this approach is not data-efficient for
high-dimensional data, although the maximum dimensionality estimate occurred at
$U = 15$ in all cases, so there may be a way to estimate dimensionality using
the shape of the curve when plotting the mean $D$ against $U$ without having
to use large amounts of data. See 
[Granata and Carnevale](https://doi.org/10.1038/srep31377) for related (but 
better) ideas.

## Conclusion

Being able to distinguish between low (say $D \leq 5$) and higher dimensionality
might still be useful under some circumstances, even when not using
multi-scaling. However, as noted by Lee and co-workers, the use of the gaussian
bandwidth compared to the "hard" cutoff of a ball with a fixed radius means that
the estimation is subject to edge effects. Therefore, in multi-scaling, only the
dimensionality averaged over all points in the dataset is used.

Up: [Index](index.html)
