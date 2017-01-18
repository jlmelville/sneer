---
title: "Symmetric Embedding with Asymmetric Kernels"
output: html_document
---

This is a tangent based on [deriving embedding gradients](gradients.html). It's
not going to make any sense unless you read that first.

### Asymmetric Kernels with Symmetric Embedding

When [deriving the gradient](gradients.html) for 
[inhomogeneous t-SNE](http://dx.doi.org/10.1007/978-3-319-46675-0_14) I 
mentioned that the pair-wise normalization that is part of 
Symmetric SNE seems to give slightly better results than the point-wise 
normalization used in Asymmetric SNE. But I also mentioned that inhomogeneous
kernel parameters could introduce a complexity for symmetric approaches.

In order to avoid outlying points with negligible gradient, the input 
probability matrix $P$ is symmetrized by setting 
$p_{i,j} = \left(p_{i|j} + p_{j|i}\right) / 2$. There's no need to do this to
the output probabilities, because they are always joint by construction.
This is so because the weight matrix is symmetric, which in turn is guaranteed
by the kernel functions, which work on symmetric transformed distances, 
$f_{ij}$. The only way to introduce asymmetry would be to parameterize the
kernel functions, but for the output kernels commonly used, they either have
no free parameters (the t-SNE kernel) or have a single global value (e.g.
setting $\beta_i = 1$ for the SSNE and ASNE kernel, or $\alpha$ in the 
heavy-tailed kernel in HSSNE).

The one exception is inhomogeneous t-SNE, which has a degree of freedom 
parameter, $\nu_i$, which is allowed to vary per point. However, in this case
inhomogeneous t-SNE is an asymmetric method, so we work with conditional output
probabilities anyway.

But what if we wanted to extend this approach to symmetric methods? In this case
we could either:

* Ignore the fact that the input probabilities are joint, and the output 
probabilities are conditional. As only one of the probability matrices are 
symmetric, I would call this a "semi-symmetric" embedding.
* Apply the same averaging to $Q$ as is done to $P$.

The first approach is pragmatic. Do we really care about the inconsistency
between $P$ and $Q$? After all, most embeddings are initialized from a Normal
distribution with a very small standard deviation, so initial distances are
all very short and hence you shouldn't see many small gradients. Hopefully
this won't lead to outlying data points with small probabilities.

But if we did want to explicitly symmetrize the output probabilities, how 
would this affect the gradient? The simplest way to do this which preserves
the structure of the current derivation is to rewrite the normalization step.

This is the old normalization:

$$q_{ij} = \frac{w_{ij}}{\sum_{kl} w_{kl}} = \frac{w_{ij}}{S}$$

We now have a new definition:
$$
q_{ij} = \frac{q_{i|j} + q_{j|i}}{2} = \frac{w_{ij} + w_{ji}}{2S}
$$

We also need a new expression for the derivative of the joint probability with
respect to the weights. Once again, we have two different expressions, depending
on whether the weight we care about is in the numerator or not.

$$
\frac{\partial q_{ij}}{\partial w_{ij}} = 
\frac{1}{2}\left[\frac{1}{S} - \frac{w_{ij}}{S^2} - \frac{w_{ji}}{S^2}\right] = 
\frac{1}{2S} - \frac{q_{i|j} + q_{j|i}}{2S}=
\frac{1}{2S} - \frac{q_{ij}}{S}
$$
and:
$$
\frac{\partial q_{kl}}{\partial w_{ij}} = 
-\frac{1}{2}\left[\frac{w_{kl}}{S^2} + \frac{w_{lk}}{S^2}\right] = 
-\frac{q_{k|l} + q_{l|k}}{2S} =
-\frac{q_{kl}}{S}
$$

So after all that, the gradient doesn't look all that different from the old
normalization. The only other thing to remember is that:

We can now find a new expression for

$$
\sum_{kl}
\frac{\partial C}{\partial q_{kl}}
\frac{\partial q_{kl}}{\partial w_{ij}}
$$

needed for calculating $k_{ij}$. Using our new equations we have:

$$
\sum_{kl}
\frac{\partial C}{\partial q_{kl}}
\frac{\partial q_{kl}}{\partial w_{ij}}
=
\frac{1}{S}
\left[
+ \frac{1}{2}\frac{\partial C}{\partial q_{ij}}
+ \frac{1}{2}\frac{\partial C}{\partial q_{ji}}
-
\sum_{kl} \frac{\partial C}{\partial q_{kl}} 
q_{kl}
\right]
$$
but given that the probabilities are now symmetric, that means that
$\partial C/\partial q_{ij} = \partial C/\partial q_{ji}$ so:

$$
\sum_{kl}
\frac{\partial C}{\partial q_{kl}}
\frac{\partial q_{kl}}{\partial w_{ij}}
=
\frac{1}{S}
\left[
\frac{\partial C}{\partial q_{ij}}
-
\sum_{kl} \frac{\partial C}{\partial q_{kl}} 
q_{kl}
\right]
$$
After all that, the plug-in gradient is therefore exactly the same. This is 
good news.

But this *doesn't* mean that we can use the simplified gradients for ASNE, SSNE,
t-SNE and so on. The simplification there relied on $q_{ij} = w_{ij} / S$, and
it's actually now the case that $q_{i|j} = w_{ij} / S$, so there are no longer
cancellations with expressions for $\partial C / \partial q_{ij}$ to take
advantage of.

So: if you want to use asymmetric kernels with symmetric embedding, keep on
using the same generic plug-in gradient you were using before. But don't re-use
specific simplified gradients given in the literature, such as those commonly
stated for t-SNE and SSNE. Some simplification may be possible, but you will
need to re-derive these.
