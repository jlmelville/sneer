---
title: "Dynamic HSSNE"
author: "James Melville"
date: "January 9, 2017"
output: html_document
---

Up: [Index](index.html)

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
uglier multiplier of $p_{ij} - q_{ij}$. You may want to pull a $1/\alpha$ term
out to give:

$$
\frac{\partial C}{\partial \alpha} = 
  \frac{1}{\alpha}
  \sum_{ij}
  \left[
    \left(
      \frac{f_{ij}}{\alpha f_{ij} +1}
      -
      \frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha}
    \right)
      \left(
        p_{ij} - q_{ij}
      \right)
\right]
$$

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
\frac{2\xi}{\alpha}\left[
\frac{f_{ij}}{\alpha f_{ij} +1}
-
\frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha}
\right]
w_{ij}
$$

$$
\frac{\partial C}{\partial \xi} = 
  \frac{2\xi}{\alpha}
  \sum_{ij}
  \left[
    \left(
      \frac{f_{ij}}{\alpha f_{ij} +1}
      -
      \frac{\ln\left(\alpha f_{ij} + 1\right)}{\alpha}
    \right)
      \left(
        p_{ij} - q_{ij}
      \right)
\right]
$$

Now, as long as you remember to convert back and forth between $\alpha$ and
$\xi$ where needed, we can now optimize $\alpha$ at the same time as the 
coordinates.

## Point-wise normalization

Good news: the asymmetric version (i.e. DHASNE) has exactly the same gradient,
just replace $p_{ij}$ and $q_{ij}$ with $p_{i|j}$ and $q_{i|j}$, respectively.
The derivation differs only in exactly the same way that the plugin gradient
derivation differs between asymmetric and symmetric versions, which was already
covered on the [gradients](gradient.html) page.

## Including Precisions

As mentioned in the [gradients](gradient.html) page, although the original
HSSNE paper doesn't include an exponential decay factor in its expression, it's
easy to add it. For completeness, instead of fixing $\beta$ to a global value,
we'll allow it to be different for each point, as is done for the input
probabilities, so we'll label it $\beta_i$:

$$w_{ij} = \frac{1}{\left(\alpha \beta_{i} f_{ij} + 1\right)^{\frac{1}{\alpha}}}$$

The effect on the gradient is minimal:

$$
\frac{\partial w_{ij}}{\partial \alpha} 
=-
\left[
\frac{\beta_i f_{ij}}{\alpha \beta_i \left(\alpha f_{ij} +1\right)}
-
\frac{\ln\left(\alpha \beta_i f_{ij} + 1\right)}{\alpha ^ 2}
\right]
w_{ij}
$$
And so, skipping to the actual derivative we care about:
$$
\frac{\partial C}{\partial \xi} = 
  \frac{2\xi}{\alpha}
  \sum_{ij}
  \left[
    \left(
      \frac{\beta_i f_{ij}}{\alpha \beta_i f_{ij} +1}
      -
      \frac{\ln\left(\alpha \beta_i f_{ij} + 1\right)}{\alpha}
    \right)
      \left(
        p_{ij} - q_{ij}
      \right)
\right]
$$

Using this version of the gradient makes DHSSNE compatible with techniques where
$\beta_i \neq 1$ (e.g. multiscaling and some versions of NeRV). Where $\beta_i$
values are allowed to differ from each other, there are some things to be
aware of when considering creating a symmetric embedding using an 
[asymmetric kernel](asymmetric-kernel-gradient.html).
The short version: you can no longer assume that $Q$ is a joint probability 
matrix by construction, and if you decided to treat $Q$ in the same way as $P$
(i.e. averaging $q_{i|j}$ and $q_{j|i}$), the gradient can't be simplified as 
much as has been shown here.

## Inhomogeneous HSSNE

What about making $\alpha$ point-wise, i.e. having one $\alpha_i$ per point?
This can also be done, and the good news is that the gradient is pretty much
the same, except we only need to sum over $j$:

$$
\frac{\partial C}{\partial \xi_i} = 
  \frac{2\xi_i}{\alpha_i}
  \sum_{j}
  \left[
    \left(
      \frac{\beta_i f_{ij}}{\alpha_i \beta_i f_{ij} +1}
      -
      \frac{\ln\left(\alpha_i \beta_i f_{ij} + 1\right)}{\alpha_i}
    \right)
      \left(
        p_{ij} - q_{ij}
      \right)
\right]
$$
We may as well call this inhomogeneous HSSNE, in analogy with inhomogeneous 
t-SNE.

## Inhomogeneous HASNE

Just as was the case with DHASNE, the gradient for inhomogeneous HASNE is the
same as for HSSNE, replacing joint probabilities with conditional probabilities.

## Optimizing the Precisions

If we want to optimize the $\beta_i$ values too, the gradient wrt to the weight
is:

$$
\frac{\partial w_{ij}}{\partial \beta_i} 
=-
f_{ij}
w_{ij}^{\alpha_i + 1}
$$

This can be inserted into the same expression we used before without any further
complications, to eventually get to:

$$
\frac{\partial C}{\partial \beta_i} = 
  \sum_{j}
    f_{ij}w_{ij}^{\alpha_{i}}
      \left(
        p_{ij} - q_{ij}
      \right)
$$

Like $\alpha$, $\beta$ cannot take non-positive values. So in practice you'd 
also transform $\beta_i$ into a variable like $\xi_i$, which only requires 
multiplying the above gradient by $2\xi_i$.

## The it-SNE gradient

To demonstrate the connection between HSSNE and it-SNE, here's the gradient
with respect to $\nu_i$, which is analogous to $\alpha_i$ in inhomogeneous
HSSNE:

$$
\frac{\partial C}{\partial \nu_i} = 
  \frac{1}{2}
  \sum_{j}
  \left\{
    \ln\left(\frac{f_{ij}}{\nu_i} + 1\right)
    -
    \left[
      \frac{f_{ij}\nu_i + 1}{\nu_i^2 \left(\frac{f_{ij}}{\nu_i} + 1\right)}
    \right]
      \left(
        p_{ij} - q_{ij}
      \right)
\right\}
$$
This has a very similar structure to the HSSNE version. The extension to the 
gradient with respect to $\xi$ is obvious (i.e. multiply the RHS in the above 
equation by $2\xi$).

Up: [Index](index.html)
