---
title: "Dynamic HSSNE"
author: "James Melville"
date: "January 9, 2017"
output: html_document
---

The original stochastic neighbor embedding approach used gaussian kernels to
produce weights in both the input and output space. The advance of t-SNE was
to use the the Student's t-distribution with one degree of freedom (also
known as the Cauchy distribution) for the output space, which allows the
output distances to be longer.

The heavy-tailed SNE approach 
([HSSNE](https://papers.nips.cc/paper/3770-heavy-tailed-symmetric-stochastic-neighbor-embedding))
generalizes SSNE and t-SNE by allowing the degree of "stretching" of the output
distances to be controlled by a heavy-tailedness parameter, $\alpha$, which
with a value of 0 gives a Gaussian and hence behaves like SSNE, and a value of 
1 gives the Cauchy distribution, and hence behaves like t-SNE:

$$w_{ij} = \frac{1}{\left(\alpha f_{ij} + 1\right)^{\frac{1}{\alpha}}}$$

The HSSNE paper gives examples where deviating from $\alpha=1$ gives better 
results, but doesn't provide any guidance on how to choose a value.

[Inhomogeneous t-SNE](http://dx.doi.org/10.1007/978-3-319-46675-0_14) suggests
a very similar scheme, but allows for differences in the density of the data
by assigning what's effectively a different heavy-tailedness parameter for
each data point. They also suggest choosing values for each of these parameters
by including them in the optimization process along with the coordinates.

Could we do the same with the global $\alpha$ parameter used in HSSNE? We could.

Assuming a pairwise normalization scheme (as used in HSSNE), and the notation
introduced in the [gradients](gradients.html) section, we can write the partial 
derivative relating the total error to $\alpha$ as:

$$
\frac{\partial C}{\partial \alpha} = 
  \sum_{ij} 
  \frac{\partial C}{\partial q_{ij}}
  \sum_{kl}
  \frac{\partial q_{ij}}{\partial w_{kl}}
  \frac{\partial w_{kl}}{\partial \alpha}
$$

Fortunately, the derivation proceeds almost exactly like it does for the 
coordinate case on the gradients page, so let's fast-forward to:

$$
\frac{\partial C}{\partial \alpha} = 
  \sum_{ij}\left[\frac{1}{S}
    \left(\frac{\partial C}{\partial q_{ij}}
      - \sum_{kl}\frac{\partial C}{\partial q_{kl}}q_{kl}
    \right)
    \frac{\partial w_{ij}}{\partial \alpha}
  \right]
$$

For HSSNE, we use the Kullback-Leibler divergence, so we simplify to:

$$
\frac{\partial C}{\partial \alpha} = 
  \sum_{ij}\left[\frac{1}{S}
    \left(
      -\frac{p_{ij}}{q_{ij}} + 1
    \right)
    \frac{\partial w_{ij}}{\partial \alpha}
  \right]
$$


The gradient of the kernel with respect to $\alpha$ is:

$$\frac{\partial w_{ij}}{\partial \alpha} =
\left[
\frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha ^ 2}
-
\frac{f_{ij}}{\alpha \left(\alpha f_{ij} +1\right)}
\right]
w_{ij}
=-
\left[
\frac{f_{ij}}{\alpha \left(\alpha f_{ij} +1\right)}
-
\frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha ^ 2}
\right]
w_{ij}
$$

where, using the power of precognition, I'm pulling out a factor of -1, which 
comes in handy below.

Inserting the expression for the gradient, we get:

$$
\frac{\partial C}{\partial \alpha} = 
  \sum_{ij}
  \left[
    \left(
      \frac{f_{ij}}{\alpha \left(\alpha f_{ij} +1\right)}
      -
      \frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha ^ 2}
    \right)
    \left(\frac{-w_{ij}}{S}\right)
      \left(
        -\frac{p_{ij}}{q_{ij}} + 1
      \right)
\right]
$$

We've already seen when deriving the t-SNE gradient that there's a 
pleasant bit of cancelling when you're using the KL divergence and a kernel
gradient that is proportional to $-w_{ij}$. And mercifully, this is still
the case:

$$
\frac{\partial C}{\partial \alpha} = 
  \sum_{ij}
  \left[
    \left(
      \frac{f_{ij}}{\alpha \left(\alpha f_{ij} +1\right)}
      -
      \frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha ^ 2}
    \right)
     \left(-q_{ij}\right)
      \left(
        -\frac{p_{ij}}{q_{ij}} + 1
      \right)
\right]
=
  \sum_{ij}
  \left[
    \left(
      \frac{f_{ij}}{\alpha \left(\alpha f_{ij} +1\right)}
      -
      \frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha ^ 2}
    \right)
      \left(
        p_{ij} - q_{ij}
      \right)
\right]
$$

This looks just like the SSNE or t-SNE gradient, except with an admittedly
uglier multiplier of $p_{ij} - q_{ij}$.

In principle, this is all you need to optimize $\alpha$ as one extra parameter.
However, there's the slight problem that $\alpha$ is constrained to be larger
than zero, and the gradient explicitly contains a log term which will blow up
if you have the temerity to allow $\alpha$ to go non-positive.

This problem also affects the free parameters in inhomogeneous t-SNE, which must
also be positive. The simple solution they use is to define a new variable,
which in terms of HSSNE, would be:

$$\alpha = \xi^2 + \epsilon$$

where $\epsilon$ is a small constant to enforce positivity (0.001 in the 
inhomogeneous t-SNE paper). The optimization is now done in terms of $\xi$,
rather than $\alpha$.

Fortunately, we didn't just waste our time coming up with the gradient in
terms of $\alpha$, as the substitution of $\xi$ for $\alpha$ in the heavy tail
expression doesn't change the gradient too much:

$$
\frac{\partial w_{ij}}{\partial \xi} 
=-
2\xi\left[
\frac{f_{ij}}{\alpha \left(\alpha f_{ij} +1\right)}
-
\frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha ^ 2}
\right]
w_{ij}
$$

$$
\frac{\partial C}{\partial \xi} = 
  2\xi
  \sum_{ij}
  \left[
    \left(
      \frac{f_{ij}}{\alpha \left(\alpha f_{ij} +1\right)}
      -
      \frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha ^ 2}
    \right)
      \left(
        p_{ij} - q_{ij}
      \right)
\right]
$$

Now, as long as you remember to convert back and forth between $\alpha$ and
$\xi$ where needed, we can now optimize $\alpha$ at the same time as the 
coordinates.
