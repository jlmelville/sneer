---
title: "Experimental Gradients"
output:
  html_document:
    theme: cosmo
    math_method: katex
    css: gradients.css
    toc: true
    toc_float:
      collapsed: false
---

Back: [Gradients](gradients.html) Up: [Index](index.html)

Gradients for literature methods that aren't currently implemented in `sneer` or weirdo ideas that
don't correspond to any current literature method, but may provide inspiration for future
directions.

## Generic Gradient

Our starting point is the pair-wise normalized gradient from the [first part](gradients.html). As
before, sums over pairs exclude self pairs, and the raw-distance derivative assumes $d_{ij}>0$.

$$
\frac{\partial C}{\partial \mathbf{y}_i} =
  \sum_{j} \left(
  k_{ij}
  +
  k_{ji}
  \right)
  \frac{\partial f_{ij}}{\partial d_{ij}}
\frac{1}{d_{ij}}\left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

with:

$$
k_{ij} =
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

Divergences are normally applied to probabilities. But they have been generalized to work with other
values. Some of this is discussed in the [ws-SNE
paper](https://proceedings.mlr.press/v32/yange14.html), using definitions by [Cichocki and
co-workers](https://doi.org/10.3390/e13010134).

We can therefore define a set of algorithms, halfway between the distance-based embeddings like
Sammon Mapping, and probability-based embeddings like SNE, that use the un-normalized weights. I
can't think of a good name for these: un-normalized embeddings? Weight-based embeddings?
Similarity-based embeddings?

Anyway, the chain of variable dependencies is
$C \rightarrow w \rightarrow f \rightarrow d \rightarrow \mathbf{y}$. We can re-use all the
derivations we calculated for probability-based embeddings, including assuming that we're going to
use squared Euclidean distances as input to the weighting function, to get to:

$$
\frac{\partial C}{\partial \mathbf{y}_i} =
  2\sum_j^N \left(
    \frac{\partial C}{\partial w_{ij}}
    \frac{\partial w_{ij}}{\partial f_{ij}}
    +
    \frac{\partial C}{\partial w_{ji}}
    \frac{\partial w_{ji}}{\partial f_{ji}}
   \right)
   \left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

Without a normalization step, we can go straight to defining:

$$
k_{ij} = \frac{\partial C}{\partial w_{ij}}
\frac{\partial w_{ij}}{\partial f_{ij}}
$$

The generalized divergences are defined in terms of the output weights, $w_{ij}$ and the input
weights, for which I've not had to come up with a symbol before now. Let's call them $v_{ij}$. The
generalized Kullback Leibler divergence (also known as the I-divergence, and apparently used in
non-negative matrix factorization) and its derivative with respect to the output weights is:

$$
C_{ij} = v_{ij}\log\left(\frac{v_{ij}}{w_{ij}}\right) - v_{ij} + w_{ij}
$$

$$
\frac{\partial C}{\partial w_{ij}} = 1 - \frac{v_{ij}}{w_{ij}}
$$

I'm not aware of any embedding algorithms that define their cost function with the un-normalized KL
divergence, although the [ws-SNE paper](https://proceedings.mlr.press/v32/yange14.html) shows that
[elastic embedding (PDF)](http://faculty.ucmerced.edu/mcarreira-perpinan/papers/icml10.pdf) can be
considered a variant of this.

With exponential output weights and symmetric input and output weights, the gradient of the
I-divergence would be:

$$
\frac{\partial C}{\partial \mathbf{y}_i} =
4\sum_j^N \left(v_{ij} - w_{ij} \right)
\left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

which as you can see turns out to be the SSNE gradient, but with un-normalized weights replacing the
normalized versions. You can also generate a t-distributed version by multiplying by $w_{ij}$ in an
entirely analogous fashion to how you'd get t-SNE from SSNE.

### Elastic Embedding

<span id="elastic_embedding"></span>

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

As given here, the input weights are not symmetric to make the derivation as generic as possible.
The original EE paper uses a global $\beta$ for the input weights so all the terms are symmetric. We
will explicitly double count identical pairs in the symmetric case and not make any corrections as
we did when discussing distance-based embedding methods.

If we define output weights as in SSNE as:

$$
w_{ij} = \exp\left(-d_{ij}^2\right)
$$

then we can rewrite the cost function in terms of the weights as:

$$
C =
-\sum_{ij} v_{ij}^{+} \log w_{ij} +
\lambda \sum_{ij} v_{ij}^{-} w_{ij}
$$

and the gradient of the cost function with respect to the weight is:

$$
\frac{\partial C}{\partial w_{ij}} =
- \frac{v_{ij}^{+}}{w_{ij}}
+\lambda v_{ij}^{-}
$$

Given our use of exponential weights, the derivative with respect to the squared distance is:

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
\frac{\partial C}{\partial \mathbf{y}_i} =
  2\sum_j^N \left[
    v_{ij}^{+} + v_{ji}^{+}
    -
    \lambda \left( v_{ij}^{-}{w_{ij}} + v_{ji}^{-}{w_{ji}} \right)
   \right]
   \left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

and if symmetric input and output weights are used:

$$
\frac{\partial C}{\partial \mathbf{y}_i} =
  4\sum_j^N \left(
    v_{ij}^{+}
    -
    \lambda v_{ij}^{-}{w_{ij}}
   \right)
   \left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

### LargeVis

[LargeVis](https://arxiv.org/abs/1602.00370) concerns itself deeply with being scalable. It does so
by partitioning the data into a set of nearest neighbors of each point, which are subject to
attractive forces between each other. All other pairs of points are repelled. Optimization occurs by
stochastic gradient descent using a batch size of one, i.e. one pair is picked at random and their
distance adjusted according to their gradient. For efficiency, neighboring and non-neighboring pairs
are sampled at different rates. I recommend close scrutiny of both the paper and the [source
code](https://github.com/lferry007/LargeVis) to get a handle on it and the exact gory details.

But if we didn't care about efficiency or partitioning the data into neighbors and non-neighbors (we
can call this method SmallVis), the cost function would look like:

$$
C =
-\sum_{ij} p_{ij} \log w_{ij}
-\gamma \sum_{ij} \log \left( 1 - w_{ij} \right)
$$

LargeVis maximizes a log-likelihood function, but here we're going to write it as a negative
log-likelihood, so it can be treated as a minimization problem like with every other cost function
we've looked at.

As you can see, the structure of this cost function is very similar to Elastic Embedding, except
LargeVis uses the t-distribution for the output weights. For this dense variant, I've normalized the
input weights as in SSNE and t-SNE. The $\gamma$ value plays a similar role to $\lambda$ in elastic
embedding, balancing attraction and repulsion. In LargeVis that balance also depends on how often
negative pairs are sampled and how they are chosen, so its recommended value of $7$ applies to its
own sampling scheme.

The gradient of the cost function with respect to the weight is:

$$
\frac{\partial C}{\partial w_{ij}} =
- \frac{p_{ij}}{w_{ij}}
+ \frac{\gamma}{1 - w_{ij}}
$$

The weight gradient for the t-distribution is:

$$
\frac{\partial w_{ij}}{\partial f_{ij}} = -w_{ij}^2
$$

leading to a force constant of

$$
k_{ij} = \frac{\partial C}{\partial w_{ij}}
\frac{\partial w_{ij}}{\partial f_{ij}}
= p_{ij} w_{ij}
-\frac{\gamma w_{ij}^{2}}{1 - w_{ij}}
= p_{ij} w_{ij}
-\frac{\gamma w_{ij}}{d_{ij}^2}
$$

The gradient is therefore:

$$
\frac{\partial C}{\partial \mathbf{y}_i} =
  4\sum_j^N \left(
    p_{ij} w_{ij}
    -\frac{\gamma w_{ij}}{d_{ij}^2}
   \right)
   \left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

I've assumed symmetric input and output weights for this dense gradient. For asymmetric weights, we
can keep both $k_{ij}$ and $k_{ji}$ as we did for elastic embedding.

### The Connection Between SNE, EE, and LargeVis

LargeVis interprets each output weight as the probability of an edge. EE uses unnormalized
similarities. Even with these different interpretations, their cost functions have much in common
with SSNE and t-SNE. Let's expand the SSNE cost function:

$$
C =
\sum_{ij} p_{ij} \log \frac{p_{ij}}{q_{ij}}
=
\sum_{ij} \left( p_{ij} \log p_{ij} - p_{ij} \log q_{ij} \right)
$$

The first term has no dependence on the output coordinates, so is a constant we'll just mark as
$C_{P}$. Now let's write out $q_{ij}$ as $w_{ij} / Z$ where $Z$ is the sum of the weights:

$$
\begin{aligned}
C &= C_P - \sum_{ij} p_{ij} \log \frac{w_{ij}}{Z} \\
&= C_P - \sum_{ij} p_{ij} \log w_{ij} + \sum_{ij} p_{ij} \log Z.
\end{aligned}
$$

After some rearranging and re-writing $Z$ back to a sum of weights:

$$
\begin{aligned}
C &= C_P - \sum_{ij} p_{ij} \log w_{ij} + \log Z \sum_{ij} p_{ij} \\
&= C_P - \sum_{ij} p_{ij} \log w_{ij} + \log \sum_{ij} w_{ij}.
\end{aligned}
$$

And let's finally express $p_{ij}$ in terms of weights:

$$
C =
C_P - \frac{1}{\sum_{kl} v_{kl}} \sum_{ij} v_{ij} \log w_{ij} + \log \sum_{ij} w_{ij}
$$

Let's review all three cost functions we went into detail on, and also throw in the generalized KL
divergence as $C_{\mathrm{I}}$, suitably rearranged in the way we just did for SSNE. Here $C_{\mathrm{LV}}$ is
the dense SmallVis variant:

$$
\begin{aligned}
C_{\mathrm{SSNE}} &=
-\frac{1}{\sum_{kl} v_{kl}} \sum_{ij} v_{ij}\log w_{ij} + \log \sum_{ij} w_{ij} + C_P, \\
C_{\mathrm{LV}} &= -\sum_{ij} p_{ij}\log w_{ij} - \gamma \sum_{ij} \log(1 - w_{ij}), \\
C_{\mathrm{EE}} &= -\sum_{ij} v_{ij}^{+}\log w_{ij} + \lambda \sum_{ij} v_{ij}^{-}w_{ij}, \\
C_{\mathrm{I}} &= -\sum_{ij} v_{ij}\log w_{ij} + \sum_{ij} w_{ij} + C_V.
\end{aligned}
$$

Leaving out the constants, each cost function has an attractive and a repulsive term. The attractive
terms agree when we use the same input weights and output kernel. SNE-like repulsion involves a log
of a sum, which makes the cost nonseparable. The repulsion terms for EE and the LargeVis variant are
sums of per-pair contributions, which makes them amenable to stochastic gradient approaches. Both
use a free parameter to balance attraction and repulsion.

### The SSNE gradient with un-normalized weights

While the cost functions are clearly similar, it's the form of the gradients that will determine the
difference between the results we see in these methods, so let's look at those.

Here's the force constant used in the gradient of SSNE, rewritten in terms of the un-normalized
weights:

$$
k_{ij} = p_{ij} - q_{ij} = \left[ \frac{v_{ij}}{S} - \frac{w_{ij}}{Z} \right]
$$

where $S$ is the sum of the input weights, $S = \sum_{ij} v_{ij}$, in analogy with $Z$. For t-SNE,
we just have to multiply both contributions by $w_{ij}$, so we'll stick with SSNE to avoid clutter.
Take a factor of $1 / S$ out of that expression and the SSNE gradient is:

$$
\frac{\partial C_{\mathrm{SSNE}}}{\partial \mathbf{y}_i} =
\frac{4}{S}
\sum_j^N \left( v_{ij} -\frac{S}{Z} w_{ij} \right)
\left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

And now let's compare that with the same cost functions we just looked at, assuming symmetric input
and output weights:

$$
\begin{aligned}
\frac{\partial C_{\mathrm{I}}}{\partial \mathbf{y}_i} &=
4\sum_j^N \left(v_{ij} - w_{ij} \right)
\left(\mathbf{y}_i - \mathbf{y}_j\right), \\
\frac{\partial C_{\mathrm{EE}}}{\partial \mathbf{y}_i} &=
4\sum_j^N \left( v_{ij}^{+} - \lambda v_{ij}^{-}{w_{ij}} \right)
\left(\mathbf{y}_i - \mathbf{y}_j\right), \\
\frac{\partial C_{\mathrm{LV}}}{\partial \mathbf{y}_i} &=
4\sum_j^N \left(p_{ij} -\frac{\gamma}{1 - w_{ij}} w_{ij} \right) w_{ij}
\left(\mathbf{y}_i - \mathbf{y}_j\right).
\end{aligned}
$$

For LargeVis, I've kept with the t-distributed weights (the other gradients use exponential
kernels), which explains the extra factor of $w_{ij}$ it has. I've also expressed $1 / d_{ij}^2$
back in terms of weights to make the connection with the other gradients clearer.

The $1 / S$ constant term that got pulled out of the SSNE gradient is uninteresting from an
optimization stand point. The important difference is that $S/Z$ term. That's the key to what
normalization does: the repulsive part of the gradient is re-weighted at each iteration, based on
the ratio of the sum of the weights in the input and output space. To compare with elastic
embedding, that's like dynamically choosing the $\lambda$ term each iteration.

Yang, Peltonen and Kaski go into much more detail about the connections between normalized and
un-normalized divergences (or separable and nonseparable) and specifically the connection between
Elastic Embedding and SNE in their paper [Optimization equivalence of divergences improves neighbor
embedding](https://proceedings.mlr.press/v32/yange14.html). And in [Majorization-Minimization for
Manifold Embedding](https://proceedings.mlr.press/v38/yang15a.html), they optimize t-SNE (in a
majorization-minimization context) by alternating optimizations with the ratio held constant,
followed by updating the ratio by treating it as a parameter to be optimized.

## Normalized Distances

Taking the opposite tack, if normalization is so important for probability-based embeddings, what
about normalizing the distances directly? In this case, we'll use $q_{ij}$ as a symbol, but it's
better to think of it as a normalized distance. The entries sum to one, and a larger distance gives
a larger $q_{ij}$. Rescaling all the coordinates by the same positive factor leaves $q_{ij}$
unchanged, so a cost based only on these normalized distances has no preferred overall scale.

At any rate, the chain of dependencies is $C \rightarrow q \rightarrow d \rightarrow \mathbf{y}$. As
we wrote out for distance-based embeddings:

$$
\frac{\partial C}{\partial \mathbf{y}_i} =
  \sum_j^N \left(
    \frac{\partial C}{\partial d_{ij}} +
    \frac{\partial C}{\partial d_{ji}}
   \right)
   \frac{1}{d_{ij}}\left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

and our next (and final) step in the chain is to define $\frac{\partial C}{\partial d_{ij}}$ in
terms of $q_{ij}$, i.e.

$$
\frac{\partial C}{\partial d_{ij}} =
\sum_k^N \sum_l^N \frac{\partial C}{\partial q_{kl}}
  \frac{\partial q_{kl}}{\partial d_{ij}}
$$

Given that we're normalizing over the distances, we once again have to decide if we want to do a
point-wise or pair-wise normalization. Like before, I'll stick with the pair-wise normalization. We
can treat $\frac{\partial q_{kl}}{\partial d_{ij}}$ exactly like we did
$\frac{\partial q_{kl}}{\partial w_{ij}}$ for probability-based embeddings, and to cut a long story
short we get to:

$$
\frac{\partial C}{\partial d_{ij}} =
\frac{1}{S} \left[
  \frac{\partial C}{\partial q_{ij}} - \sum_{kl} \frac{\partial C}{\partial q_{kl}} q_{kl}
\right]
$$

Here $q_{ij} = d_{ij}/S$ and $S = \sum_{kl} d_{kl}$. As a quick check, take $C=\sum_{kl} q_{kl}=1$.
Every $\partial C/\partial q_{kl}$ is one, so the two terms in brackets cancel.

With pair-wise normalization, symmetric distances and a cost that treats $q_{ij}$ and $q_{ji}$
symmetrically give equal distance derivatives. The coordinate gradient then becomes:

$$
\frac{\partial C}{\partial \mathbf{y}_i} =
  2\sum_j^N k_{ij} \left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

and

$$
k_{ij} = \frac{1}{d_{ij}S} \left[
  \frac{\partial C}{\partial q_{ij}} - \sum_{kl} \frac{\partial C}{\partial q_{kl}} q_{kl}
\right]
$$

Point-wise normalization generally gives different derivatives in the two directions, so keep both
contributions in that case.

We can now plug in metric MDS or Sammon-like cost derivatives with normalized distances in place of
the raw distances in the cost.

## Transformed Distances

We could also consider keeping with raw distances and their square loss, but transforming them. In
this case, the chain of variable dependencies is
$C \rightarrow f \rightarrow d \rightarrow \mathbf{y}$.

Once again, let's start with:

$$
\frac{\partial C}{\partial \mathbf{y}_i} =
  \sum_j^N \left(
    \frac{\partial C}{\partial d_{ij}} +
    \frac{\partial C}{\partial d_{ji}}
   \right)
   \frac{1}{d_{ij}}\left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

and now we use the chain rule to connect $\frac{\partial C}{\partial d_{ij}}$ with $f_{ij}$. We
don't need to worry about any terms except the $ij$th one:

$$
\frac{\partial C}{\partial d_{ij}} =
  \frac{\partial C}{\partial f_{ij}}
  \frac{\partial f_{ij}}{\partial d_{ij}}
$$

When we use the SNE-style transformation:

$$
f_{ij} = d_{ij}^2
$$

$$
\frac{\partial f_{ij}}{\partial d_{ij}} = 2d_{ij}
$$

and the gradient becomes:

$$
\frac{\partial C}{\partial \mathbf{y}_i} =
  2\sum_j^N \left(
    \frac{\partial C}{\partial f_{ij}} +
    \frac{\partial C}{\partial f_{ji}}
   \right)
   \left(\mathbf{y}_i - \mathbf{y}_j\right)
$$

If we use a square loss, introducing $g_{ij}$ for the input space transformed distances:

$$
C_{ij} = \left(g_{ij} - f_{ij}\right)^2
$$

$$
\frac{\partial C}{\partial f_{ij}} = -2\left(g_{ij} - f_{ij}\right)
$$

Taking $g_{ij} = r_{ij}^2$ and counting each pair once gives the [SSTRESS
criterion](gradients.html#sstress) from the first part.

Back: [Gradients](gradients.html) Up: [Index](index.html)
