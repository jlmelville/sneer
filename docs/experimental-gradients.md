---
title: "Experimental Gradients"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---

Back: [Gradients](gradients.html) Up: [Index](index.html)

Weirdo gradients that don't correspond to any current literature method, but
may provide inspiration for future directions for `sneer`.

## Generic Gradient

Our starting point is the generic embedding gradient:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_{j} \left(
  k_{ij}
  +
  k_{ji}
  \right)
  \frac{\partial f_{ij}}{\partial d_{ij}}
\frac{1}{d_{ij}}\left(\mathbf{y_i} - \mathbf{y_j}\right)
$$

with:

$$k_{ij} = 
\frac{1}{S}
\left[
\frac{\partial C}{\partial q_{ij}}
-
\sum_{kl} \frac{\partial C}{\partial q_{kl}} 
q_{kl}
\right]
\frac{\partial w_{ij}}{\partial f_{ij}}
$$

## Generalized Divergences

Divergences are normally applied to probabilities. But they have been 
generalized to work with other values. Some of this is discussed in the 
[ws-SNE paper](http://jmlr.org/proceedings/papers/v32/yange14.html), 
using definitions by 
[Cichocki and co-workers](https://dx.doi.org/10.3390/e13010134).

We can therefore define a set of algorithms, halfway between the distance-based 
embeddings like Sammon Mapping, and probability-based embeddings like SNE, that
use the un-normalized weights. I can't think of a good name for these: 
un-normalized embeddings? Weight-based embeddings? Similarity-based embeddings?

Anyway, the chain of variable dependencies is 
$C \rightarrow w \rightarrow f \rightarrow d \rightarrow \mathbf{y}$. We can
re-use all the derivations we calculated for probability-based embeddings,
including assuming that we're going to use squared Euclidean distances as input
to the weighting function, to get to:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2\sum_j^N \left(
    \frac{\partial C}{\partial w_{ij}} 
    \frac{\partial w_{ij}}{\partial f_{ij}} 
    +
    \frac{\partial C}{\partial w_{ji}} 
    \frac{\partial w_{ji}}{\partial f_{ji}} 
   \right) 
   \left(\mathbf{y_i - y_j}\right)
$$

Without any probabilities involved, we can go straight to defining:

$$
k_{ij} = \frac{\partial C}{\partial w_{ij}}
\frac{\partial w_{ij}}{\partial f_{ij}}
$$

The generalized divergences are defined in terms of the output weights, 
$w_{ij}$ and the input weights, for which I've not had to come up with a symbol 
before now. Let's call them $v_{ij}$. The generalized Kullback Leibler 
divergence (also known as the I-divergence, and apparently used in non-negative
matrix factorization) and its derivative with respect to the output weights is:

$$C_{ij} = v_{ij}\ln\left(\frac{v_{ij}}{w_{ij}}\right) - v_{ij} + w_{ij}$$
$$\frac{\partial C}{\partial w_{ij}} = 1 - \frac{v_{ij}}{w_{ij}}$$

I'm not aware of any embedding algorithms that define their cost function with 
the un-normalized KL divergence, although the 
[ws-SNE paper](http://jmlr.org/proceedings/papers/v32/yange14.html) shows that 
[elastic embedding](http://faculty.ucmerced.edu/mcarreira-perpinan/papers/icml10.pdf)
can be considered a variant of this.

### Elastic Embedding

The elastic embedding cost function is:

$$
C = 
\sum_{ij} v_{ij}^{+} d_{ij}^{2} + 
\lambda \sum_{ij} v_{ij}^{-} \exp\left(-d_{ij}^{2}\right)
$$
where the positive and negative weights are:

$$
v_{ij}^{+} = \exp\left(-\beta_i r_{ij}^{2}\right)
$$
i.e. the usual exponential input weights, and

$$
v_{ij}^{-} = r_{ij}^{2}
$$
As given here, the input weights are not symmetric to make the derivation as 
generic as possible. The original EE paper uses a global $\beta$ for the input 
weights so all the terms are symmetric.

If we define output weights as in SSNE as:

$$
w_{ij} = \exp\left(-d_{ij}^2\right)
$$

then we can rewrite the cost function in terms of the weights as:

$$
C = 
-\sum_{ij} v_{ij}^{+} \ln w_{ij} + 
\lambda \sum_{ij} v_{ij}^{-} w_{ij}
$$

and the gradient of the cost function with respect to the weight is:
$$
\frac{\partial C}{\partial w_{ij}} = 
- \frac{v_{ij}^{+}}{w_{ij}}
+\lambda v_{ij}^{-}
$$

Given our use of exponential weights, the derivative with respect to the 
squared distance is:

$$
\frac{\partial w_{ij}}{\partial f_{ij}} = -w_{ij}
$$

Thus, the force constant is:

$$
k_{ij} = \frac{\partial C}{\partial w_{ij}}
\frac{\partial w_{ij}}{\partial f_{ij}}
= v_{ij}^{+}
-\lambda v_{ij}^{-}{w_{ij}}
$$

and gradient being:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  2\sum_j^N \left[
    v_{ij}^{+} + v_{ji}^{+}
    -
    \lambda \left( v_{ij}^{-}{w_{ij}} + v_{ji}^{-}{w_{ji}} \right)
   \right]
   \left(\mathbf{y_i - y_j}\right)
$$
and if symmetric input and output weights are used:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  4\sum_j^N \left(
    v_{ij}^{+}
    -
    \lambda v_{ij}^{-}{w_{ij}}
   \right) 
   \left(\mathbf{y_i - y_j}\right)
$$

## Normalized Distances

Taking the opposite tack, if normalization is so important for 
probability-based embeddings, what about normalizing the distances directly?
In this case, we'll use $q_{ij}$ as a symbol, but it's better to think of it
as a normalized distance, even though the sum of the distances will indeed
sum to one. However, because the distances haven't been converted to 
similarities, a large distances leads to a large $q_{ij}$, which is the opposite
of how we've been thinking about things up until now.

At any rate, the chain of dependencies is 
$C \rightarrow q \rightarrow d \rightarrow \mathbf{y}$. As we wrote out for
distance-based embeddings:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N \left(
    \frac{\partial C}{\partial d_{ij}} +
    \frac{\partial C}{\partial d_{ji}}
   \right) 
   \frac{1}{d_{ij}}\left(\mathbf{y_i - y_j}\right)
$$

and our next (and final) step in the chain is to define 
$\frac{\partial C}{\partial d_{ij}}$ in terms of $q_{ij}$, i.e.

$$\frac{\partial C}{\partial d_{ij}} = 
\sum_k^N \sum_l^N \frac{\partial C}{\partial q_{kl}} 
  \frac{\partial q_{kl}}{\partial d_{ij}}$$


Given that we're normalizing over the distances, we once again have to decide
if we want to do a point-wise or pair-wise normalization. Like before, I'll 
stick with the pair-wise normalization. We can treat 
$\frac{\partial q_{kl}}{\partial d_{ij}}$ exactly like we did 
$\frac{\partial q_{kl}}{\partial w_{ij}}$ for probability-based embeddings, and
to cut a long story short we get to:

$$\frac{\partial C}{\partial d_{ij}} = 
-\frac{1}{S} 
  \left[ 
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl} + 
      \frac{\partial C}{\partial q_{ij}} 
  \right]
$$

where $S$ is now the sum of $d_{ij}$. If we employ pair-wise normalization, 
then despite the normalization being present, with no weighting function 
involved, we can still be sure that 
$\frac{\partial C}{\partial d_{ij}} = \frac{\partial C}{\partial d_{ji}}$, and
the gradient can be written as:
$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2\sum_j^N k_{ij} \left(\mathbf{y_i - y_j}\right)
$$
and
$$k_{ij} = 
-\frac{1}{d_{ij}S} 
  \left[ 
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl} + 
      \frac{\partial C}{\partial q_{ij}} 
  \right]
$$

However, if you're doing point-wise normalization, there is an asymmetry, so
there's no further simplification.

We now have can be plugged in with metric MDS and Sammon-like cost function 
derivatives, but with the distances replaced by normalized distances.

## Transformed Distances

We could also consider keeping with raw distances and their square loss, but
transforming them. In this case, the chain of variable dependencies is 
$C \rightarrow f \rightarrow d \rightarrow \mathbf{y}$. 

Once again, let's start with:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N \left(
    \frac{\partial C}{\partial d_{ij}} +
    \frac{\partial C}{\partial d_{ji}}
   \right) 
   \frac{1}{d_{ij}}\left(\mathbf{y_i - y_j}\right)
$$

and now we use the chain rule to connect $\frac{\partial C}{\partial d_{ij}}$
with $f_{ij}$. We don't need to worry about any terms except the $ij$th one:

$$\frac{\partial C}{\partial d_{ij}} = 
  \frac{\partial C}{\partial f_{ij}}
  \frac{\partial f_{ij}}{\partial d_{ij}}$$

When we use the SNE-style transformation:

$$f_{ij} = d_{ij}^2$$
$$\frac{\partial f_{ij}}{\partial d_{ij}} = 2d_{ij}$$

and the gradient becomes:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2\sum_j^N \left(
    \frac{\partial C}{\partial f_{ij}} +
    \frac{\partial C}{\partial f_{ji}}
   \right) 
   \left(\mathbf{y_i - y_j}\right)
$$

If we use a square loss, introducing $g_{ij}$ for the input space transformed
distances:

$$C_{ij} = \left(g_{ij} - f_{ij}\right)^2$$
$$\frac{\partial C}{\partial f_{ij}} = -2\left(g_{ij} - f_{ij}\right)$$

Back: [Gradients](gradients.html) Up: [Index](index.html)

