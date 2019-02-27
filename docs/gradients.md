---
title: "Deriving Embedding Gradients"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---

Next: [Experimental Gradients](experimental-gradients.html) Up: [Index](index.html)

There are lots of different approaches to deriving the t-SNE gradient and 
related methods in the literature. The one that helped me the most is the
one given by Lee and co-workers in the 
[JSE paper](https://dx.doi.org/10.1016/j.neucom.2012.12.036).

The following discussion attempts to do something similar, but taking things
even slower. Also, I am a lot less rigorous and I've tried to use less notation,
but to also be more generic, avoiding making any assumptions about the cost
function or weighting function.

The only mathematical abilities you should need for this is the ability to do 
basic partial differentiation and the chain rule for partial derivatives,
which happens to be:

## Chain rule for partial derivatives

Say we have a function $x$, of $N$ variables $y_1, y_2 \dots y_i \dots y_N$, and
each $y$ is a function of $M$ variables $z_1, z_2, \dots z_j \dots z_M$, then 
the partial derivative of $x$ with respect to one of $z$ is:

$$\frac{\partial x}{\partial z_j} = 
  \sum_i^N \frac{\partial x}{\partial y_i}\frac{\partial y_i}{\partial z_j}
$$

Also, if this isn't obvious, the various commutative, associative and 
distributive properties of addition and multiplication mean that we can move
around nested sigma notation:

$$\sum_i x_i \sum_j y_j \equiv \sum_j y_j \sum_i x_i 
\equiv \sum_i \sum_j x_i y_j \equiv \sum_{ij} x_i y_j
$$

There are going to be a lot of nested summations at the beginning of the 
derivation, so I will be making use of that final short hand where multiple 
summation indices indicate a nested summation.

## Notation

I assume you are familiar with the basic [concepts](concepts.html) of the 
approach of SNE and related methods. I'll use the following notation:

* $\mathbf{y_i}$ is the $i$th embedded coordinate in the lower dimension.
* $P$ is the matrix of input probabilities, $Q$ is the matrix of output 
probabilities.
* $p_{ij}$ means the $\left(i, j\right)$th element of the matrix $P$.
* I'll also use $i$, $j$, $k$ and $l$ as indexes into various matrices.
* The number of points being embedded is $N$.

I'll assume that there is an input probability matrix already created, and that
the cost function involves $P$ and $Q$ and hence so does the gradient. At each
stage of the optimization we need to get from the current set of coordinates
$\mathbf{y_1}, \mathbf{y_2} \dots \mathbf{y_i} \dots \mathbf{y_N}$, to a 
gradient matrix. The procedure is as follows:

* Create the distance matrix, where the element $d_{ij}$ represents the distance
between point $i$ and $j$.
* Transform the distances to create $f_{ij}$.
* Apply a weighting function to create a weight, $w_{ij}$, such that the larger
the weight, the smaller the distance between $i$ and $j$. Because of this 
inverse relationship between the weight and the distance, I will refer to this
weight as a similarity as it makes it easier to remember that a big weight
refers to a small distance.
* Convert the weights into a probability, $q_{ij}$. This achieved by normalizing
over a sum of weights. There are two approaches to defining this sum, which
affects the interpretation of the probability. See below for more on this.

Once the output probabilities $q_{ij}$ are calculated, you are generally in 
possession of enough data to calcuate the gradient, with the exact form
depending on the nature of the cost and similarity function.

Before going further, let's look at the two approaches to probability 
calculation.

## Symmetric vs asymmetric embedding

### Point-wise probabilities

The original SNE approach converted the weights into probabilities by:

$$q_{ij} = \frac{w_{ij}}{\sum_k^N w_{ik}}$$

That is, we consider all similarities involving point $i$. Let's call this the
point-wise approach. A consequence of this is that $q_{ij} \neq q_{ji}$ and
hence this results in an asymmetric probability matrix, $Q$. In fact, (at least
in the `sneer` implementation), each row of the matrix is a separate probability 
distribution, where each row sums to one. In the point-wise approach you are
calculating $N$ different divergences, with each point being responsible for
a separate probability distribution.

The point-wise normalization to create $N$ probabilities is the scheme used
in what is now called Asymmetric SNE.

### Pair-wise probabilities

Another way to convert the weights is:

$$q_{ij} = \frac{w_{ij}}{\sum_k^N \sum_l^N w_{kl}}$$

This normalizes by using _all_ pairs of points, so we'll call this the pair-wise
approach (in the 
[ws-SNE paper](http://jmlr.org/proceedings/papers/v32/yange14.html), it's 
referred to as "matrix-wise"). The resulting matrix $Q$ contains a single 
probability distribution, i.e. the grand sum of the matrix is one. Using this 
normalization, it's still true that, in general, $q_{ij} \neq q_{ji}$, but when 
creating the input probability matrix $P$, $p_{ij}$ and $p_{ji}$ are averaged 
so that they are equal to each other. In the case of the output weights, the 
function that generates them  always produces symmetric weights, so that 
$w_{ij} = w_{ji}$ which naturally leads to $q_{ij} = q_{ji}$. The resulting 
matrix is therefore symmetric without having to do any extra work.

This pair-wise scheme is used in what is called Symmetric SNE and t-distributed
SNE.

Obviously these two schemes are very similar to each other, but it's easy to
get confused when looking at how different embedding methods are implemented.
As to whether it makes much of a practical difference, in the 
[JSE paper](https://dx.doi.org/10.1016/j.neucom.2012.12.036) Lee and 
co-workers say that it has "no significant effect" on JSE, whereas in the 
[t-SNE paper](http://www.jmlr.org/papers/v9/vandermaaten08a.html), 
van der Maaten and Hinton note that SSNE sometimes produced results that were 
"a little better" than ASNE. Not a ringing endorsement either way, but in my 
experiments with `sneer`, the symmetrized (pair-wise) normalization seems to 
produce better results.

Also, to avoid taking up too much space, from now on, I will omit the $N$ from
the upper range of the summation, and where there are nested summations, I will
only use one $\sum$ symbol, with the number of summation indices below the
sigma indicating whether it's nested, i.e.

$$q_{ij} 
= \frac{w_{ij}}{\sum_k^N \sum_l^N w_{kl}}
\equiv \frac{w_{ij}}{\sum_{kl} w_{kl}}
$$

## Breaking down the cost function

With all that out of the way, let's try and define the gradient with respect
to the cost function.

We'll start by defining a chain of dependent variables specifically for 
probability-based embeddings. A glance at the chain rule for partial 
derivatives above indicates that we're going to be using a lot of nested 
summations of multiple terms, although mercifully most of them evaluate to 0 
and disappear. But for now, let's ignore the exact indices. To recap the 
variables we need to include and the order of their dependencies:

* The cost function, $C$ is normally a divergence of some kind, and hence 
expressed in terms of the output probabilities, $q$.
* The output probabilities, $q$, are normalized versions of the similarity
weights, $w$.
* The similarity weights are generated from a function of the distances, $f$.
* The $f$ values are a function of the Euclidean distances, $d$. Normally,
this is the squared distance.
* The distances are generated from the coordinates, $\mathbf{y_i}$.

We're going to chain those individual bits together via the chain rule
for partial derivatives. The chain of variable dependencies is 
$C \rightarrow q
\rightarrow w \rightarrow f \rightarrow d \rightarrow \mathbf{y}$.

### Gradient with pair-wise normalization

We can write the partial derivative relating the total error
to a coordinate $h$ as:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{ij} 
  \frac{\partial C}{\partial q_{ij}}
  \sum_{kl}
  \frac{\partial q_{ij}}{\partial w_{kl}}
  \sum_{mn}
  \frac{\partial w_{kl}}{\partial f_{mn}}
  \sum_{pq}
  \frac{\partial f_{mn}}{\partial d_{pq}}
  \frac{\partial d_{pq}}{\partial \mathbf{y_h}}  
$$

I told you there would be a lot of nested summations. Let's make some of them
disappear. 

The relationship between $q$ and $w$ depends on whether we are using a 
point-wise or pair-wise normalization. For now, let's assume a pair-wise
normalization. The difference is not enormous, so let's come back to it later.
No matter which normalization you use, the good news is that the relationship 
between $w$, $f$ and $d$ is such that any cross terms are 0, i.e. unless 
$k=m=p$ and $l=n=q$ those derivatives evaluate to 0, which immediately gets us 
to:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{ij} 
  \frac{\partial C}{\partial q_{ij}}
  \sum_{kl}
  \frac{\partial q_{ij}}{\partial w_{kl}}
  \frac{\partial w_{kl}}{\partial f_{kl}}
  \frac{\partial f_{kl}}{\partial d_{kl}}
  \frac{\partial d_{kl}}{\partial \mathbf{y_h}}  
$$

Also, either $k=h$ or $l=h$, otherwise 
$\partial d_{kl}/\partial \mathbf{y_h}=0$ which leads to:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{ij} 
  \frac{\partial C}{\partial q_{ij}}
  \sum_{l}
  \frac{\partial q_{ij}}{\partial w_{hl}}
  \frac{\partial w_{hl}}{\partial f_{hl}}
  \frac{\partial f_{hl}}{\partial d_{hl}}
  \frac{\partial d_{hl}}{\partial \mathbf{y_h}}  
+
  \sum_{ij} 
  \frac{\partial C}{\partial q_{ij}}
  \sum_{k}
  \frac{\partial q_{ij}}{\partial w_{kh}}
  \frac{\partial w_{kh}}{\partial f_{kh}}
  \frac{\partial f_{kh}}{\partial d_{kh}}
  \frac{\partial d_{kh}}{\partial \mathbf{y_h}}
$$
Now to rearrange some of the grouping of the summations:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{l} \left(
  \sum_{ij} 
  \frac{\partial C}{\partial q_{ij}}
  \frac{\partial q_{ij}}{\partial w_{hl}}
  \right)
  \frac{\partial w_{hl}}{\partial f_{hl}}
  \frac{\partial f_{hl}}{\partial d_{hl}}
  \frac{\partial d_{hl}}{\partial \mathbf{y_h}}  
+
  \sum_{k} \left(
  \sum_{ij} 
  \frac{\partial C}{\partial q_{ij}}
  \frac{\partial q_{ij}}{\partial w_{kh}}
  \right)
  \frac{\partial w_{kh}}{\partial f_{kh}}
  \frac{\partial f_{kh}}{\partial d_{kh}}
  \frac{\partial d_{kh}}{\partial \mathbf{y_h}}
$$
At this point we can rename some of the indices: $h$ becomes $i$, $i$ becomes
$k$, $j$ becomes $l$ and $k$ and $l$ become $j$. This gives:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_{j} \left(
  \sum_{kl} 
  \frac{\partial C}{\partial q_{kl}}
  \frac{\partial q_{kl}}{\partial w_{ij}}
  \right)
  \frac{\partial w_{ij}}{\partial f_{ij}}
  \frac{\partial f_{ij}}{\partial d_{ij}}
  \frac{\partial d_{ij}}{\partial \mathbf{y_i}}  
+
  \sum_{j} \left(
  \sum_{kl} 
  \frac{\partial C}{\partial q_{kl}}
  \frac{\partial q_{kl}}{\partial w_{ji}}
  \right)
  \frac{\partial w_{ji}}{\partial f_{ji}}
  \frac{\partial f_{ji}}{\partial d_{ji}}
  \frac{\partial d_{ji}}{\partial \mathbf{y_i}}
$$

Also, let's assume that both the distances and transformed distances will
be symmetric, $d_{ij} = d_{ji}$ and $f_{ij} = f_{ji}$. The former is certainly
true if we're using the mathematical concept of a metric as a distance (and I'm
unaware of anyone using anything other than the Euclidean distance anyway), and
the latter can be made so by construction, by just moving any parameterization
that would lead to asymmetric transformed distances into the weight calculation.

This lets us write:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_{j} \left[
  \left(
  \sum_{kl} 
  \frac{\partial C}{\partial q_{kl}}
  \frac{\partial q_{kl}}{\partial w_{ij}}
  \right)
  \frac{\partial w_{ij}}{\partial f_{ij}}
+
  \left(
  \sum_{kl} 
  \frac{\partial C}{\partial q_{kl}}
  \frac{\partial q_{kl}}{\partial w_{ji}}
  \right)
  \frac{\partial w_{ji}}{\partial f_{ji}}
  \right]
  \frac{\partial f_{ij}}{\partial d_{ij}}
  \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
$$

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_{j} \left(
  k_{ij}
  +
  k_{ji}
  \right)
  \frac{\partial f_{ij}}{\partial d_{ij}}
  \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
$$

with

$$k_{ij} = 
\left[
\sum_{kl}
\frac{\partial C}{\partial q_{kl}}
\frac{\partial q_{kl}}{\partial w_{ij}}
\right]
\frac{\partial w_{ij}}{\partial f_{ij}}
$$

Without getting into any specifics of the functional form of the cost function
or weighting function, there's a further simplification we can make. The 
probabilities are always created by a straightforward normalization of the 
weights. We're considering just the pair-wise normalization for now, and as 
we've seen, the relationship between $w$ and $q$ is:

$$q_{ij} = \frac{w_{ij}}{\sum_{kl} w_{kl}} = \frac{w_{ij}}{S}$$

but now we've introduced $S$, the sum of all the weights involving all pairs.

It's important to realize that any particular weight, $w_{ij}$, appears in both 
the expression for its equivalent probability, $q_{ij}$ (where it appears in 
the numerator and denonimator) _and_ in the expression for all the other 
probabilities, $q_{kl}$, where $i \neq k$ and $j \neq l$. In the latter case, it 
appears only in the denominator, but it still leads to non-zero derivatives.

Thus, we have two forms of the derivative to consider:
$$\frac{\partial q_{ij}}{\partial w_{ij}} = \frac{S - w_{ij}}{S^2} = 
  \frac{1}{S} - \frac{q_{ij}}{S}
$$
and:
$$\frac{\partial q_{kl}}{\partial w_{ij}} = 
  -\frac{w_{kl}}{S^2} = 
  -\frac{q_{kl}}{S}
$$

They're nearly the same expression, there's just one extra $1/S$ term 
to consider when $i=k$ and $j=l$.

Inserting these expressions into the one we had for $k_{ij}$:

$$
k_{ij} = 
\left[
\sum_{kl}
\frac{\partial C}{\partial q_{kl}}
\frac{\partial q_{kl}}{\partial w_{ij}}
\right]
\frac{\partial w_{ij}}{\partial f_{ij}}
=
\frac{1}{S}
\left[
\frac{\partial C}{\partial q_{ij}}
-
\sum_{kl} \frac{\partial C}{\partial q_{kl}} 
q_{kl}
\right]
\frac{\partial w_{ij}}{\partial f_{ij}}
$$

This is far as we can get with $k_{ij}$ without choosing a cost and weighting
function, but we can simplify the distance part of the expression if we're
prepared to assume that we're only going to want to use Euclidean distances
in the output.

While there may be some exotic situations where the output distances should be
non-Euclidean (a literary analysis of HP Lovecraft perhaps, or more seriously,
the use of 
[hyperbolic distances to model hierarchical data](https://arxiv.org/abs/1705.08039)), 
the t-SNE literature always has $d_{ij}$ represent Euclidean distances. In a
$K$-dimensional output space, the distance between point $\mathbf{y_i}$ and
point $\mathbf{y_j}$ is:

$$d_{ij} = \left[\sum_l^K\left (y_{il} - y_{jl} \right )^2\right ]^{1/2}$$

and the derivative can be written as:

$$\frac{\partial d_{ij}}{\partial \mathbf{y_i}} = 
\frac{1}{d_{ij}}\left(\mathbf{y_i} - \mathbf{y_j}\right)
$$

Therefore, a completely generic expression for the gradient is:

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

with

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

If you want to get a tiny bit more specific, I am also unaware of any 
t-SNE-related methods that don't transform the distances by simply squaring
them:

$$f_{ij} = d_{ij}^{2}$$

Allow me to insult your intelligence by writing out the gradient:

$$\frac{\partial f_{ij}}{\partial d_{ij}} = 2d_{ij}$$

which cleans things up even more:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  2 \sum_{j} \left(
  k_{ij}
  +
  k_{ji}
  \right)
\left(\mathbf{y_i} - \mathbf{y_j}\right)
$$
This is now looking more like the expected "points on springs" interpretation of
the gradient, with the $k_{ij}$ representing the force constant (stiffness) of 
each spring, and $\mathbf{y_i - y_j}$ the displacement.

The above equation is useful because as long as you can define the gradient
of a cost function in terms of $q$ and the gradient of a similarity kernel in
terms of $f$, you can mix and match these terms and get the gradient of the
cost function with respect to the embedded coordinates without too much trouble,
which is all you need to optimize the coordinates with a standard gradient
descent algorithm.

### Gradient with point-wise normalization

The above expression was derived for the pair-wise normalization. I said we'd
get back to the point-wise version and now seems like a good time. The 
derivation proceeds as normal initially, where we cancel out all the cross 
terms associated with $w$, $f$ and $d$:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{ij} 
  \frac{\partial C}{\partial q_{ij}}
  \sum_{kl}
  \frac{\partial q_{ij}}{\partial w_{kl}}
  \frac{\partial w_{kl}}{\partial f_{kl}}
  \frac{\partial f_{kl}}{\partial d_{kl}}
  \frac{\partial d_{kl}}{\partial \mathbf{y_h}}  
$$

There is now one extra simplification we can make. Let's refresh our memories
over the form of the point-wise normalization:

$$q_{ij} = \frac{w_{ij}}{\sum_k w_{ik}}$$

i.e. it's only over weights associated with point $i$. In terms of cancelling
cross terms in the derivative, we can see that unless $i = k$, then
$\partial q_{ij}/\partial w_{kl} = 0$, so we can now replace $k$ with $i$:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{ij} 
  \frac{\partial C}{\partial q_{ij}}
  \sum_{l}
  \frac{\partial q_{ij}}{\partial w_{il}}
  \frac{\partial w_{il}}{\partial f_{il}}
  \frac{\partial f_{il}}{\partial d_{il}}
  \frac{\partial d_{il}}{\partial \mathbf{y_h}}  
$$

The derivation then proceeds just as before, where $h=l$ or $h=i$, and we
end up at the same gradient expression, but with a minor change to $k_{ij}$:

$$
k_{ij} =
\left[
\sum_{k}
\frac{\partial C}{\partial q_{ik}}
\frac{\partial q_{ik}}{\partial w_{ij}}
\right]
\frac{\partial w_{ij}}{\partial f_{ij}}
$$

We will still expand $\partial q_{ik}/\partial w_{ij}$, also involving a slight 
change in notation for $q_{ij}$:

$$q_{ij} = \frac{w_{ij}}{\sum_k w_{ik}} = \frac{w_{ij}}{S_i}$$

i.e. we now have to give an $i$ subscript to $S$, because the sum of the weights 
only includes those associated with point $i$. The partial derivatives are:

$$\frac{\partial q_{ij}}{\partial w_{ij}} = \frac{1}{S_i} - \frac{q_{ij}}{S_i}
$$
and:
$$\frac{\partial q_{ik}}{\partial w_{ij}} = -\frac{q_{ik}}{S_i}$$

and inserting this into $k_{ij}$ gives:

$$k_{ij} = 
\frac{1}{S_i}
\left[
\frac{\partial C}{\partial q_{ij}}
-
\sum_{k} \frac{\partial C}{\partial q_{ik}} 
q_{ik}
\right]
\frac{\partial w_{ij}}{\partial f_{ij}}
$$

Comparing this with the pair-wise version, you can see they are very similar.

### Summary

After all that, we can summarise with three equations. The probability-based 
embedding gradient is:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  2 \sum_{j} \left(
  k_{ij}
  +
  k_{ji}
  \right)
\left(\mathbf{y_i} - \mathbf{y_j}\right)
$$

where for a point-wise normalization:
$$k_{ij} = 
\frac{1}{S_i}
\left[
\frac{\partial C}{\partial q_{ij}}
-
\sum_{k} \frac{\partial C}{\partial q_{ik}} 
q_{ik}
\right]
\frac{\partial w_{ij}}{\partial f_{ij}}
$$

and for a pair-wise normalization:
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

Plug whatever combination of cost function derivative and weight function
derivative you like, and off you go.

## Some Cost Functions and their Derivatives

### Kullback-Leibler Divergence

Used by the ASNE, SSNE, t-SNE and variants:

$$C = D_{KL}(P||Q) = \sum_{ij} p_{ij}\log\left(\frac{p_{ij}}{q_{ij}}\right)$$
$$\frac{\partial C}{\partial q_{ij}} = - \frac{p_{ij}}{q_{ij}}$$

### "Reverse" Kullback-Leibler Divergence

[NeRV](http://www.jmlr.org/papers/v11/venna10a.html) symmetrizes the KL 
divergence by also considering the cost when $q_{ij}$ is the "reference" 
probability distribution:

$$C = D_{KL}(Q||P) = \sum_{ij} q_{ij}\log\left(\frac{q_{ij}}{p_{ij}}\right)$$
$$\frac{\partial C}{\partial q_{ij}} = \log\left(\frac{q_{ij}}{p_{ij}}\right) + 1$$

### Jensen-Shannon Divergence

The Jensen-Shannon Divergence, as defined in 
[JSE](https://dx.doi.org/10.1016/j.neucom.2012.12.036) is:

$$C = D_{JS}(P||Q) = 
\frac{1}{1-\kappa}D_{KL}(P||Z) + 
\frac{1}{\kappa}D_{KL}(Q||Z)
$$
where $Z$ is a mixture of P and Q:
$$Z = \kappa P + \left(1-\kappa \right)Q$$

Rather than invoke the chain rule to couple $Z$ to $Q$, I'm just going to write
out the derivatives for the two parts of the JS divergence with respect to $Q$
(ignoring the $\kappa$ weighting for now):

$$\frac{\partial D_{KL}(P||Z)}{\partial q_{ij}} = 
- \left(1 - \kappa \right) \frac{p_{ij}}{z_{ij}}
$$
$$
\frac{\partial D_{KL}(Q||Z)}{\partial q_{ij}} = 
\kappa \left(\frac{p_{ij}}{z_{ij}}\right) 
- \log\left(\frac{q_{ij}}{z_{ij}}\right)
$$

Once you add these derivatives together, multiplying by the $\kappa$ values in
the cost function, terms cancel to leave a surprisingly simple derivative:

$$
\frac{\partial C}{\partial q_{ij}} = 
\frac{\partial D_{JS}(p_{ij}||q_{ij})}{\partial q_{ij}} =
-\frac{1}{\kappa}\log\left(\frac{q_{ij}}{z_{ij}}\right)
$$

### $\chi^2$ (Chi-square):

[Im, Verma and Branson](https://arxiv.org/abs/1811.01247) describe replacing the
KL divergence with other members of the f-divergence family. These include
the reverse KL divergence (which we looked at as part of NeRV) and the Jensen-Shannon
divergence we just looked at in JSE. But there are other members of the f-divergence
family. One is the $\chi^2$ (chi-square) divergence:

$$C = \sum_{ij} \frac{\left(p_{ij} - q_{ij}\right)^2}{q_{ij}^2}$$

$$\frac{\partial C}{\partial q_{ij}} = -\frac{p_{ij}^2 - q_{ij}^2}{q_{ij}^2} = -\frac{p_{ij}^2}{q_{ij}^2} + 1$$

### Hellinger Distance

The other member of the f-divergences that Im and co-workers look at is the
Hellinger distance:

$$C = \sum_{ij} \left(\sqrt{p_{ij}} - \sqrt{q_{ij}}\right)^2$$

$$\frac{\partial C}{\partial q_{ij}} = -\frac{\sqrt{p_{ij}} - \sqrt{q_{ij}}}{\sqrt{q_{ij}}} = -\frac{\sqrt{p_{ij}}}{\sqrt{q_{ij}}} + 1$$

The gradient for the chi-square and Hellinger distance versions of t-SNE aren't
stated, so we will revisit these divergences when deriving some force constants
for more complex divergences below.

### $\alpha$-$\beta$ Divergence

[Narayan, Punjani and Abbeel](http://proceedings.mlr.press/v37/narayan15.html) 
describe AB-SNE, which uses the alpha-beta divergence., where $\alpha$ and
$\beta$ are free parameters.

$$C =\frac{1}{\alpha \beta} \sum_{ij} -p_{ij}^{\alpha} q_{ij}^{\beta} + \frac{\alpha}{\alpha + \beta}p_{ij}^{\alpha + \beta} + \frac{\beta}{\alpha + \beta}q_{ij}^{\alpha + \beta}$$

$$\frac{\partial C}{\partial q_{ij}} = -\frac{1}{\alpha} q_{ij}^{\beta - 1} \left( p_{ij}^{\alpha} - q_{ij}^{\alpha} \right)$$

It's not very clear from the cost function, but when we derive the gradient
below, you can see that you get t-SNE back with $\alpha = 1$ and $\beta = 0$

## Some Similarity Kernels and their Derivatives

### Exponential/Gaussian Kernel

The weighting function most commonly encountered is a gaussian kernel, although
because I've separated the distance transformation from the weighting, it's 
actually exponential with respect to the transformed distances:

$$w_{ij} = \exp\left(-\beta_{i} f_{ij}\right)$$

Also, I've thrown in the exponential decay factor, $\beta_{i}$ which is used in
the [Input Initialization](input-initialization.html) for the input kernel
only, as part of the perplexity-based calibration (aka entropic affinities). 
As discussed further in that section, the output kernel function normally
just sets all the $\beta$ values to one, so it effectively disappears from
the output kernel gradient. However, some methods do make use of an output
$\beta$, so here is the general gradient:

$$\frac{\partial w_{ij}}{\partial f_{ij}} 
= -\beta_{i} \exp\left(-\beta_{i} f_{ij}\right)
= -\beta_{i} w_{ij}
$$

### t-Distribution Kernel

As used in the output kernel of t-SNE. Also referred to as the Cauchy 
distribution sometimes.

$$w_{ij} = \frac{1}{1 + f_{ij}}$$
$$\frac{\partial w_{ij}}{\partial f_{ij}} 
= -\frac{1}{\left(1 + f_{ij}\right)^2}
= -w_{ij}^2
$$

### Heavy-tailed Kernel

A generalization of the exponential and t-distributed kernel, and described in
the 
[HSSNE paper](https://papers.nips.cc/paper/3770-heavy-tailed-symmetric-stochastic-neighbor-embedding).

$$w_{ij} = \frac{1}{\left(1 + \alpha \beta_{i} f_{ij} \right)^{1 / \alpha}}$$
$$\frac{\partial w_{ij}}{\partial f_{ij}} 
= - \frac{\beta_{i}}{\left(1 + \alpha \beta_{i} f_{ij}\right)^{\left(\alpha + 1\right) / \alpha}}
= -\beta_{i} w_{ij} ^ \left(\alpha + 1\right)
$$

The degree of heavy-tailedness is controlled by $\alpha$: when $\alpha = 1$, the
kernel behaves like the t-distributed kernel. As it approaches 0, it approaches
the exponential kernel. $\alpha > 1$ allows output distances to stretch even
more than t-SNE. I am unaware of any research that looks at seeing if there is
a way to find an optimal value of $\alpha$ for a given dataset, although the
inhomogeneous t-SNE method mentioned below solves the problem by optimizing
a similar set of values along with the embedded coordinates.

I have also included the precision parameter $\beta$ in the equation. This isn't
present in the original HSSNE paper, but it's fairly easy to do the integration 
to work out how to include it. This shouldn't be taken as a suggestion to use
it as an input kernel - heavy-tailedness can make it very easy for it to be
impossible to get lower values of perplexity. However, it does allow for an 
output kernel that behaves like t-SNE, but has the free precision parameter 
that would allow t-SNE to be used with the method described in 
[multiscale JSE](https://dx.doi.org/10.1016/j.neucom.2014.12.095). See the
sections on [Input Initialization](input-initialization.html) and
[Embedding Methods](embedding-methods.html) for how to do that in `sneer`.

Other groups have looked at heavy-tailed kernels with slightly different 
definitions of the heavy-tailedness parameter, which I will continue to represent
as $\alpha$. In [parametric t-SNE](http://proceedings.mlr.press/v5/maaten09a), 
van der Maaten used a general t-distribution:

$$w_{ij} = \frac{1}{\left(1 + f_{ij} / \alpha \right)^{\left(\alpha + 1\right)/2}}$$

["twice" t-SNE (tt-SNE)](http://hdl.handle.net/2078.1/200844) uses:

$$w_{ij} = \frac{1}{\left(1 + f_{ij} / \alpha \right)^{\alpha / 2}}$$

[Kobak, Linderman and co-workers](https://arxiv.org/abs/1902.05804)
defined the kernel as:

$$w_{ij} = \frac{1}{\left(1 + f_{ij} / \alpha \right)^\alpha}$$

### Inhomogeneous t-SNE

[Inhomogeneous t-SNE](http://dx.doi.org/10.1007/978-3-319-46675-0_14) defines 
yet another heavy-tailed kernel, using nearly the same definition as in
parametric t-SNE. I'll swap to the notation that this paper uses for the degrees
of freedom, $\nu$:

$$w_{ij} = \left(1 + \frac{f_{ij}}{\nu_{i}}\right)^{-\left(\nu_{i} + 1\right)/2}$$

where $\nu_{i} = \infty$ gives SNE-like behavior and $\nu_{i} = 1$ gives t-SNE
behavior. The twist here is that as the $i$ subscript indicates, the degrees of
freedom is allowed to vary per-point, rather than the global value HSSNE uses.

The derivative is:

$$\frac{\partial w_{ij}}{\partial f_{ij}} 
= - \frac{\nu_{i} + 1}{2\left(f_{ij} + \nu_{i}\right)}w_{ij}
$$
Adding inhomogeneity to the kernel opens up a lot of potential, but there is a
potential complexity to think about when using this an 
[asymmetric kernel](asymmetric-kernel-gradient.html) with the pair-wise 
normalization method. We'll not worry about it, because first of all, you can
keep on using the same plug-in gradient we already derived (but not the
simplified versions we'll derive in the next section), and second, I am unaware
of any literature method that combines these two ideas: inhomogeneous t-SNE,
despite its name, actually uses a point-wise normalization, making it more like
a t-Distributed version of ASNE, rather than SSNE. For point-wise normalization
schemes the complications don't arise.

### Importance Weighted Kernel

Yang, Peltonen and Kaski described 
[weighted t-SNE](http://jmlr.org/proceedings/papers/v32/yange14.html) (wt-SNE), 
in which the output weights, $w_{ij}$ are are modified to $m_{ij} w_{ij}$, where 
$m_{ij}$ indicates the "importance" of points $i$ and $j$. The importance was
defined as the product:

$$
m_{ij} = \deg_i \deg_j
$$

where $\deg_i$ is the degree centrality associated with a node $i$, after
interpreting the input weights as edges in a graph. There are a variety of ways
to define the degree of a node from an affinity matrix, but Yang and co-workers
use $p_{ij}$ as the degree weight, and the degree centrality of node $i$ is the 
sum of column $i$ (although as $P$ is symmetric, you could use the row sums 
too).

The weighted SSNE kernel is:

$$
w_{ij} = m_{ij} \exp\left(-\beta_{i} f_{ij}\right)
$$
and the gradient is:
$$
\frac{\partial w_{ij}}{\partial f_{ij}} 
= -\beta_{i} m_{ij} \exp\left(-\beta_{i} f_{ij}\right)
= -\beta_{i} w_{ij}
$$

which means the gradient expression for weighted SSNE doesn't change. But
doing the same for the Cauchy kernel for weighted t-SNE:

$$
w_{ij} = \frac{m_{ij}}{1 + d_{ij}^2}
$$

gives the following gradient:

$$
\frac{\partial w_{ij}}{\partial f_{ij}} 
= \frac{-m_{ij}} {\left(1 + f_{ij}\right)^2}
= \frac{-w_{ij}^2}{m_{ij}}
$$

compared to the t-SNE gradient (which we will derive below), the wt-SNE gradient
has a factor of $w_{ij} / m_{ij}$ rather than $w_{ij}$.

### g-SNE

[Zhou and Sharpee](https://doi.org/10.1101/331611) describe "global t-SNE"
(g-SNE) which, reminiscent of NeRV, uses a weighted mixture of two KL
divergences:

$$C = \sum_{ij} p_{ij} \log \frac{p_{ij}}{q_{ij}} + \lambda \sum_{ij} \hat{p}_{ij} \log \frac{\hat{p_{ij}}}{\hat{q}_{ij}}$$

The first KL divergence is the t-SNE cost. The second set of probabilities with 
hats over the quantities are associated with probabilities that are sensitive
to long range distances, rather than local similarities. The similarity kernel
used is just the reciprocal of the t-SNE kernel:

$$\hat{w}_{ij} = 1 + f_{ij} = \frac{1}{w_{ij}}$$

This has a nice and easy derivative:

$$\frac{\partial w_{ij}}{\partial f_{ij}} = 1$$

## Deriving the ASNE, SSNE and t-SNE Gradient

With a master equation and an expression for the derivative of the cost function
and kernel function, we have all we need to mix and match various costs and
kernels to our heart's content. But it would be nice to see the familiar SNE
and t-SNE gradient fall out of that mixture.

Plenty of space on this web page, though. Let's do it. This is the master
equation again:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j 
  \left(
    k_{ij} + k_{ji}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

where for a pair-wise normalization (as used by SSNE and t-SNE):
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

### Inserting the Kullback-Leibler Divergence

Time to simplify the expression for $k_{ij}$. Both SNE and t-SNE use the 
Kullback-Leibler Divergence, which as noted above, has the following
gradient:

$$\frac{\partial C}{\partial q_{ij}} = - \frac{p_{ij}}{q_{ij}}$$

$k_{ij}$ therefore becomes:

$$k_{ij} =
\frac{1}{S}
  \left[ 
  -\frac{p_{ij}}{q_{ij}} -
    \sum_{kl} 
      -\frac{p_{kl}}{q_{kl}} q_{kl}
  \right]
  \frac{\partial w_{ij}}{\partial f_{ij}} =
\frac{1}{S} 
\left[
  -\frac{p_{ij}}{q_{ij}} +
  \sum_{kl} p_{kl}
\right]
\frac{\partial w_{ij}}{\partial f_{ij}} =
\frac{1}{S} 
\left[
  -\frac{p_{ij}}{q_{ij}} +
  1
\right]
\frac{\partial w_{ij}}{\partial f_{ij}}
$$

At this point, notice that both the SNE and t-SNE output kernel (Gaussian and
t-Distribution respectively), have a derivative that has the general form

$$
\frac{\partial w_{ij}}{\partial f_{ij}} = -w_{ij}^n
$$

where $n = 1$ in the case of SNE, and $n = 2$ for t-SNE. Substituting that in, 
we now get:

$$
k_{ij} =
\frac{1}{S} 
\left[
  -\frac{p_{ij}}{q_{ij}} +
  1
\right]
\frac{\partial w_{ij}}{\partial f_{ij}} = 
-\frac{w_{ij}^n}{S} 
\left[
  -\frac{p_{ij}}{q_{ij}} +
  1
\right] = 
-w_{ij}^{n-1}q_{ij}
\left[
  -\frac{p_{ij}}{q_{ij}} +
  1
\right]
$$

using the fact that $w_{ij}/S = q_{ij}$. Now, we can move $-q_{ij}$ 
inside the expression in parentheses to get:

$$
k_{ij} =
w_{ij}^{n-1}
\left(
  {p_{ij}} - {q_{ij}}
\right)
$$

At this point, we refer back to the master equation:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j
  \left(
    k_{ij} + k_{ji}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

### ASNE Gradient (More or Less)

For SNE, $w_{ij}^{n-1} = 1$ because $n = 1$ and we get:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j 
  \left(
    p_{ij} - q_{ij} + p_{ji} - q_{ji}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$
This looks a lot like the ASNE gradient. Note, however, that ASNE used a 
point-wise normalization, so we would need to use the point-wise version of
$k_{ij}$. Fortunately, you get to the same expression by nearly the exact same 
steps, so I leave this as an exercise to you, dear reader.

### SSNE Gradient

For SSNE, there are further simplifications to be made. Both the $P$ and $Q$ 
matrices are symmetric, so $p_{ij} = p_{ji}$, and $q_{ij} = q_{ji}$, leading to:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j 
  \left(
    p_{ij} - q_{ij} + p_{ji} - q_{ji}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right) =
  2
  \sum_j 
  \left(
    2 p_{ij} - 2 q_{ij}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right) =
  4
  \sum_j 
  \left(
    p_{ij} - q_{ij}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

The familiar SSNE gradient. 

### t-SNE Gradient

For t-SNE, we get:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j 
  \left[
    w_{ij}\left(p_{ij} - q_{ij}\right) + w_{ji}\left(p_{ji} - q_{ji}\right)
  \right]
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

but the t-distributed kernel also creates a symmetric $W$ matrix so we can 
still simplify to:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  4
  \sum_j 
  w_{ij}
  \left(
    p_{ij} - q_{ij}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

Also familiar. 

## Derivation of Some Force Constants

I think we all deserve a long lie down now. But before we do, for the sake of
completeness, let's see if we can get the force constants ($k_{ij}$) out for the
slightly more complicated NeRV and JSE cost functions, as well as some methods
where the authors didn't write out the gradients. The gradients are pretty
obvious once you have $k_{ij}$, but aren't as compact as the SNE ones, so we'll
stop at $k_{ij}$.

### NeRV Force Constant

NeRV mixes two divergences, the "forward" KL divergence and the "reverse" KL
divergence, with the degree of mixing controlled by the $\lambda$ parameter:

$$
C_{NeRV} = \lambda C_{fwd} + \left(1 - \lambda \right) C_{rev} = \\
\lambda \sum_{ij} p_{ij} \log \left( \frac{p_{ij}}{q_{ij}} \right)
+
\left(1 - \lambda \right) \sum_{ij} q_{ij} \log \left( \frac{q_{ij}}{p_{ij}} \right)
$$
NeRV uses point-wise normalization, so really we should be writing $p_{ij}$
as $p_{j|i}$ but it just clutters the notation further, so I'm not going to.


Let's consider the two parts separately. First, the "forward" part of NeRV:

$$\frac{\partial C_{fwd}}{\partial q_{ij}} = - \lambda \frac{p_{ij}}{q_{ij}}$$

This is the standard KL divergence we've spent a lot of time deriving the SNE
family of gradients, and seeing as NeRV uses exponential kernel and point-wise
normalization as in ASNE, we can fast-forward straight to:

$$
k_{fwd,ij} =
\lambda
\left(
  {p_{ij}} - {q_{ij}}
\right)
$$
Now for the "reverse" part of the cost function:

$$
\frac{\partial C_{rev}}{\partial q_{ij}} = 
\left(1 - \lambda \right)
\left[
\log \left(\frac{q_{ij}}{p_{ij}}\right) + 1
\right]
$$
For exponential kernels with point-based normalization, the expression for
$k_{rev, ij}$ is:

$$
k_{rev, ij} = 
-q_{ij}
\left[
\frac{\partial C}{\partial q_{ij}}
-
\sum_{k} \frac{\partial C}{\partial q_{ik}} 
q_{ik}
\right]
$$

so inserting the expression for the cost function gradient:

$$
k_{rev, ij} = 
-\left(1 - \lambda\right) q_{ij}
\left[
\log
\left( \frac{q_{ij}}{p_{ij}} \right) + 1
-
\sum_{k} \log \left(\frac{q_{ik}}{p_{ik}}\right) q_{ik}
-
\sum_{k} q_{ik}
\right]
$$
Because $\sum_{k} q_{ik} = 1$, the second and fourth term in the main 
part of the expression cancel out, and the third term is just the reverse KL 
divergence, leaving:

$$
k_{rev, ij} = 
-\left(1 - \lambda\right) q_{ij}
\left[
\log
\left( \frac{q_{ij}}{p_{ij}} \right)
-
D_{KL}(Q||P)
\right]
$$

We can slightly simplify by multiplying in the minus sign into the main
expression, because $-\log \left( q_{ij} / p_{ij} \right) = \log \left( p_{ij} / q_{ij} \right)$,
so we end up with:

$$
k_{rev, ij} = 
\left(1 - \lambda\right) q_{ij}
\left[
\log
\left( \frac{p_{ij}}{q_{ij}} \right)
+
D_{KL}(Q||P)
\right]
$$

and the final force constant for NeRV is:

$$
k_{ij} =
\lambda
\left(
  {p_{ij}} - {q_{ij}}
\right)
+
\left(1 - \lambda\right) q_{ij}
\left[
\log
\left( \frac{p_{ij}}{q_{ij}} \right)
+
D_{KL}(Q||P)
\right]
$$

### JSE Force Constant

The JSE cost is also a mixture of forward and reverse divergences, but its
gradient with respect to the output probabilities is a lot simpler than NeRV's:

$$
\frac{\partial C}{\partial q_{ij}} =
-\frac{1}{\kappa} \log \left( \frac{q_{ij}}{z_{ij}} \right)
$$
That's in the same form as the reverse part of NeRV. Like NeRV, JSE also uses
an exponential kernel and point-wise normalization, so we can pretty much
just swap $z$ in everywhere we see $p$ in the force constant expression for
NeRV, and remember to use $1 / \kappa$ as the weighting factor:

$$
k_{ij} = 
\frac{q_{ij}}{\kappa}
\left[
\log
\left( \frac{z_{ij}}{q_{ij}} \right)
+
D_{KL}(Q||Z)
\right]
$$

If you have access to the 
[JSE paper](https://dx.doi.org/10.1016/j.neucom.2012.12.036), you can see that 
this is equivalent (once you've translated the nomenclature) to equation 37 in 
that publication. Success!

### Chi-square Force Constant

Im and co-workers don't state the gradient when using the chi-square 
divergence, so let's derive the force constant.

For this and the next few examples we will be using the t-SNE kernel. A generic
expression for $k_{ij}$ under those circumstances that we will start from 
each time is:

$$k_{ij} = -\left[ \frac{\partial C}{\partial q_{ij}} - \sum_{kl} \frac{\partial C}{\partial q_{kl}} q_{kl} \right] q_{ij}w_{ij}$$

Moving on to the chi-square divergence:

$$C = \sum_{ij} \frac{\left(p_{ij} - q_{ij}\right)^2}{q_{ij}^2}$$

and the derivative is:

$$\frac{\partial C}{\partial q_{ij}} = -\frac{p_{ij}^2 - q_{ij}^2}{q_{ij}^2} = -\frac{p_{ij}^2}{q_{ij}^2} + 1$$

Substituting into the expression for the force constant above, we can't actually
simplify all that much:

$$k_{ij} = -\left[ -\frac{p_{ij}^2}{q_{ij}^2} + 1 - \sum_{kl} \left( -\frac{p_{kl}^2}{q_{kl}} + q_{kl} \right) \right] q_{ij}w_{ij} = \left[ \frac{p_{ij}^2}{q_{ij}^2} - \sum_{kl} \frac{p_{kl}^2}{q_{kl}} \right] q_{ij}w_{ij}$$

at least we have it written down now.

### Hellinger Force Constant

We can also do the Hellinger Distance:

$$C = \sum_{ij} \left(\sqrt{p_{ij}} - \sqrt{q_{ij}}\right)^2$$

with derivative:

$$\frac{\partial C}{\partial q_{ij}} = -\frac{\sqrt{p_{ij}} - \sqrt{q_{ij}}}{\sqrt{q_{ij}}} = -\frac{\sqrt{p_{ij}}}{\sqrt{q_{ij}}} + 1$$

Again, we won't be getting much simplification, but here is the force constant.

$$k_{ij} = -\left[ -\frac{\sqrt{p_{ij}}}{\sqrt{q_{ij}}} + 1 - \sum_{kl} \left( -\sqrt{p_{ij}}\sqrt{q_{ij}} + q_{kl} \right) \right] q_{ij}w_{ij} = \left[ \frac{\sqrt{p_{ij}}}{\sqrt{q_{ij}}} - \sum_{kl} \sqrt{p_{kl}q_{kl}} \right] q_{ij}w_{ij}$$

### AB-SNE Force Constant

The AB-divergence gradient *is* given in the AB-SNE paper, but I thought it 
would be good practice to see if we can derive it. Here goes, starting with
the divergence:

$$C =\frac{1}{\alpha \beta} \sum_{ij} -p_{ij}^{\alpha} q_{ij}^{\beta} + \frac{\alpha}{\alpha + \beta}p_{ij}^{\alpha + \beta} + \frac{\beta}{\alpha + \beta}q_{ij}^{\alpha + \beta}$$

and the derivative:

$$\frac{\partial C}{\partial q_{ij}} = -\frac{1}{\alpha} q_{ij}^{\beta - 1} \left( p_{ij}^{\alpha} - q_{ij}^{\alpha} \right)$$

Not much hope of any neat cancellation here either, but here goes the
substitution into the $k_{ij}$ expression:

$$k_{ij} = -\left[ -q_{ij}^{\beta - 1} \left( p_{ij}^{\alpha} - q_{ij}^{\alpha} \right) + \sum_{kl} q_{kl}^{\beta} \left( p_{kl}^{\alpha} - q_{kl}^{\alpha} \right) \right] \frac{q_{ij}w_{ij}}{\alpha}$$

Finally, we can expand and rearrange a little to get the form given in the paper:

$$k_{ij} = \left[ p_{ij}^{\alpha} q_{ij}^{\beta - 1} - q_{ij}^{\alpha + \beta - 1} - \sum_{kl} p_{kl}^{\alpha} q_{kl}^{\beta} + \sum_{kl} q_{kl}^{\alpha + \beta} \right] \frac{q_{ij}w_{ij}}{\alpha}$$

Except I have this extra $1 / \alpha$ term in the force constant, which doesn't
show up in their gradient expression (and I think should). I have compared
analytical and finite difference gradients for a variety of settings of 
$\alpha$ and $\beta$ and the expression I have above seems correct. Doesn't
really matter, as this sort of constant is easily accounted for in the learning
rate of the optimizer. Anyway, unlike the cost function, it's easier to see that
this force constant reduces to t-SNE when $\alpha = 1, \beta = 0$.

### g-SNE Force Constant

A final derivation which is quite straightforward. Despite using a mixture of
two divergences and two separate kernels, the g-SNE gradient is very simple
compared to JSE and NeRV.

You may recall the g-SNE cost function is:

$$C = \sum_{ij} p_{ij} \log \frac{p_{ij}}{q_{ij}} + \lambda \sum_{ij} \hat{p}_{ij} \log \frac{\hat{p_{ij}}}{\hat{q}_{ij}}$$

with $\lambda$ being a weighting factor as in NeRV, the un-hatted part of the
cost function being the same as t-SNE, and the hatted quantities referring to
using the reciprocal of the usual t-SNE kernel:

$$\hat{w}_{ij} = 1 + f_{ij} = \frac{1}{w_{ij}}$$

with the derivative being:

$$\frac{\partial w_{ij}}{\partial f_{ij}} = 1$$

Ignoring $\lambda$ for the moment, the corresponding expression for
$\hat{k}_{ij}$ is therefore:

$$\hat{k}_{ij} = \left( -\frac{\hat{p}_{ij}}{\hat{q}_{ij}} + 1\right)\frac{1}{\hat{S}} = -\left( \hat{p}_{ij} - \hat{q}_{ij} \right)\frac{1}{\hat{q}_{ij}\hat{S}}$$

Because $\hat{q}_{ij} = \hat{w}_{ij} / \hat{S}$ the term outside the parentheses
cancels to give:

$$\hat{k}_{ij} = -\left( \hat{p}_{ij} - \hat{q}_{ij} \right)\frac{1}{\hat{w}_{ij}} = -\left( \hat{p}_{ij} - \hat{q}_{ij} \right)w_{ij}$$

We can combine this with the typical t-SNE expression to give an overall 
expression for g-SNE:

$$k_{ij} = \left[\left(p_{ij} - q_{ij}\right) -\lambda \left( \hat{p}_{ij} - \hat{q}_{ij} \right)\right]w_{ij}$$

A very simple modification of the t-SNE gradient.

*Now* we may take that long lie down.

## Distance-based embedding

For completeness, here's the derivation of gradients for distance-based
embeddings, such as metric MDS and Sammon Maps. The gradients for these
are well known, so if we can't derive them, we're in big trouble.

An embedding where the cost function is written in terms of the distances
is delightfully straightforward compared to what we just went through. The
chain of variable dependencies is $C \rightarrow d \rightarrow \mathbf{y}$.

Using the same chain rule like before:

$$\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{ij}
  \frac{\partial C}{\partial d_{ij}}
  \frac{\partial d_{ij}}{\partial \mathbf{y_h}}
$$
$\partial d_{ij}/\partial \mathbf{y_h}=0$ unless either $i=h$ or $j=h$:

$$
\frac{\partial C}{\partial \mathbf{y_h}} = 
  \sum_{j}
  \frac{\partial C}{\partial d_{hj}}
  \frac{\partial d_{hj}}{\partial \mathbf{y_h}}
+
  \sum_{i}
  \frac{\partial C}{\partial d_{ih}}
  \frac{\partial d_{ih}}{\partial \mathbf{y_h}}
$$
Renaming the indices $h$ to $i$ and $i$ to $j$:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_{j}
  \frac{\partial C}{\partial d_{ij}}
  \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
+
  \sum_{j}
  \frac{\partial C}{\partial d_{ji}}
  \frac{\partial d_{ji}}{\partial \mathbf{y_i}}
$$

Assuming symmetric distances, $d_{ij}=d_{ji}$:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  2\sum_{j}
  \frac{\partial C}{\partial d_{ij}}
  \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
$$

and let's go ahead and further assume Euclidean distances:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2\sum_{j}
  \frac{\partial C}{\partial d_{ij}}
  \frac{1}{d_{ij}}\left(\mathbf{y_i - y_j}\right)
$$

which makes our master equation for distance-based embeddings:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  2\sum_j k_{ij} \left(\mathbf{y_i - y_j}\right)
$$
where
$$k_{ij} = \frac{\partial C}{\partial d_{ij}}\frac{1}{d_{ij}}$$

## Distance-based Costs

Let's run over some distance-based cost functions. Because we're dealing with 
symmetric distances means that normally you'll see the cost function written
like:

$$C = \sum_{i<j} f\left(r_{ij}, d_{ij}\right)$$

where $r_{ij}$ is the distance in the input space and $f\left(r_{ij},
d_{ij}\right)$ just means some cost function that can be decomposed into a sum
of pair-wise contributions, and summing over $i<j$ so that only $d_{ij}$
contributes to the cost, and not $d_{ji}$ to avoid double counting the same
contribution.

To be consistent with the probability-based costs I'm going to write them as
full double sums and then multiply by half to take care of the double-counting:

$$C = \frac{1}{2} \sum_{ij} f\left(r_{ij}, d_{ij}\right)$$

### Metric MDS

For a standard metric MDS, the cost for a pair is just the square loss between 
the input distances $r_{ij}$ and the output distances $d_{ij}$:

$$C = \frac{1}{2} \sum_{ij} \left(r_{ij} - d_{ij}\right)^2$$

$$\frac{\partial C}{\partial d_{ij}} = -\left(r_{ij} - d_{ij}\right)$$

Plugging this into our expression for $k_{ij}$, we get:

$$k_{ij} = -\frac{\left(r_{ij} - d_{ij}\right)}{d_{ij}}$$

And then the gradient is:
$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  -2\sum_j \frac{\left(r_{ij} - d_{ij}\right)}{d_{ij}}
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

### SSTRESS

The SSTRESS criterion is

$$C = \frac{1}{2} \sum_{ij} \left(r_{ij}^2 - d_{ij}^2\right)^2$$
and:

$$\frac{\partial C}{\partial d_{ij}} = -2d_{ij}\left(r_{ij}^2 - d_{ij}^2\right)$$

The stiffness is therefore:
$$k_{ij} = -2\left(r_{ij}^2 - d_{ij}^2\right)$$

and the gradient is:
$$\frac{\partial C}{\partial \mathbf{y_i}}
  =
  -4\sum_j \left(r_{ij}^2 - d_{ij}^2\right) \left(\mathbf{y_i - y_j}\right)
$$

There is some interesting discussion of artefactual structure that can result 
from the use of the SSTRESS criterion in a paper by 
[Hughes and Lowe](https://papers.nips.cc/paper/2239-artefactual-structure-from-least-squares-multidimensional-scaling).

### Sammon Mapping

Sammon Mapping is very similar to metric MDS, except that it tries to put
more weight on preserving short distances (and hence local structure), by
weighting the cost function by $1/r_{ij}$:

$$C = \frac{1}{\sum_{ij} r_{ij}} \sum_{ij} \frac{\left(r_{ij} - d_{ij}\right)^2}{r_{ij}}$$
The first term in that expression is a normalization term. There's no factor of
half in the Sammon Stress when summing over all pairs, because if we were
correcting for double counting in the second term, we must also do so in the
denominator of the normalization term. They therefore cancel out.

The derivative is:

$$\frac{\partial C}{\partial d_{ij}} = 
   -\frac{2}{\sum_{ij} r_{ij}} \frac{\left(r_{ij} - d_{ij}\right)}{r_{ij}}
$$

Therefore:

$$k_{ij} = -\frac{2}{\sum_{ij} r_{ij}} \frac{\left(r_{ij} - d_{ij}\right)}{r_{ij}d_{ij}}$$

and the gradient is:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  -\frac{4}{\sum_{ij} r_{ij}}
  \sum_j \frac{\left(r_{ij} - d_{ij}\right)}{r_{ij}d_{ij}}
  \left(
   \mathbf{y_i - y_j}
  \right)
$$
A cursory examination of this expression, compared to how the gradient is
normally written (e.g. in 
[Sammon's original paper](https://dx.doi.org/10.1109/T-C.1969.222678)), the 
constant term in the numerator of the first part of the expression seems too 
large by a factor of 2 (i.e. you'll normally see a $-2$, not a $-4$). If you
bear in mind that:

$$ \frac{4}{\sum_{ij} r_{ij}} = \frac{2}{\sum_{i<j} r_{ij}}$$
the gradients are in fact equivalent.

If you've been able to follow along with this, you surely are now practiced 
enough to derive gradients for more experimental embedding methods. The
next section gives some examples.

Next: [Experimental Gradients](experimental-gradients.html) Up: [Index](index.html)
