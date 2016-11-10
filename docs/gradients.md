---
title: "Deriving Embedding Gradients"
output: html_document
---

Next: [Experimental Gradients](experimental-gradients.html) Up: [Index](index.html)

There are lots of different approaches to deriving the t-SNE gradient and 
related methods in the literature. The one that helped me the most is the
one given by Lee and co-workers in the 
[JSE paper](https://dx.doi.org/10.1016/j.neucom.2012.12.036).

The following discussion attempts to do something similar, but taking things
even slower. Also, I am a lot less rigorous and I've tried to use less notation.

The only mathematical ability you should need for this is the ability to do 
basic partial differentiation, and know the chain rule for partial derivatives,
which happens to be:

## Chain rule for partial derivatives

Say we have a function $x$, of $N$ variables $y_1, y_2 \dots y_i \dots y_N$, and
each $y$ is a function of $M$ variables $z_1, z_2, \dots z_j \dots z_M$, then 
the partial derivative of $x$ with respect to one of $z$ is:

$$\frac{\partial x}{\partial z_j} = 
  \sum_i^N \frac{\partial x}{\partial y_i}\frac{\partial y_i}{\partial z_j}$$

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
in the sneer implementation), each row of the matrix is a separate probability 
matrix, where each row sums to one. In the point-wise approach you are
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
creating the input probabilities $p_{ij}$, $p_{ij}$ and $p_{ji}$ are averaged 
so that they are equal to each other. In the case of the output weights, the 
function that generates them  always produces symmetric weights, so that 
$w_{ij} = w_{ji}$ which naturally leads to $q_{ij} = q_{ji}$, so the resulting 
matrix is symmetric without having to do any extra work.

This pair-wise scheme is used in what is called Symmetric SNE and t-distributed
SNE.

Obviously these two schemes are very similar to each other, but it's easy to
get confused when looking at how different embedding methods are implemented.
As to whether it makes much of a practical difference, in the 
[JSE paper]((https://dx.doi.org/10.1016/j.neucom.2012.12.036) Lee and 
co-workers say that it has "no significant effect" on JSE, whereas in the 
[t-SNE paper](http://www.jmlr.org/papers/v9/vandermaaten08a.html), 
van der Maaten and Hinton note that SSNE sometimes produced results that were 
"a little better" than ASNE. Not a ringing endorsement either way, but in my 
experiments with `sneer`, the symmetrized (pair-wise) normalization seems to 
produce better results.

## Breaking down the cost function

With all that out of the way, let's try and define the gradient. We'll start by 
defining a chain of dependent variables specifically for probability-based 
embeddings. A glance at the chain rule for partial derivatives above indicates 
that we're going to be using a lot of nested summations of multiple terms, 
although mercifully most of them evaluate to 0 and disappear. But for now, 
let's ignore the exact indexes. To recap the variables we need to include and
the order of their dependencies:

* The cost function, $C$ is normally a divergence of some kind, and hence 
expressed in terms of the output probabilities, $q$.
* The output probabilities, $q$, are normalized versions of the similarity
weights, $w$.
* The similarity weights are generated from a function of the distances, $f$.
* The $f$ values are a function of the Euclidean distances, $d$. Normally,
this is the squared distance.
* The distances are generated from the coordinates, $\mathbf{y_i}$.

We're going to chain those individual bits together via the chain rule
for partial derivatives. The chain of variable dependencies is $C \rightarrow q
\rightarrow w \rightarrow f \rightarrow d \rightarrow \mathbf{y}$.

I find the best way to proceed is to start by writing out the gradient with 
respect to $\mathbf{y}$ in terms of the distances, then proceeding backwards to
$q$ until we have a product of simple expressions that can have their 
derivatives easily calculated.

Some of these terms are often varied by different researchers to produce 
different types of embedding methods, e.g. the cost with respect to the 
probability (the divergence) or the similarity weighting kernel (e.g. gaussian
or t-distribution). Other parts are never changed (e.g. the output distance
function, how weights are converted to probabilities). Where there is universal
agreement, I will explicitly write out the function and its derivative. For the
functions which are often changed, I'll leave them generic. 

### Distance, $d_{ij}$

To start, let's consider $C$, $d$ and $\mathbf{y}$. Using the chain rule we can
write out the gradient of the cost function with respect to the $i$th embedded 
point as:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N \sum_k^N \frac{\partial C}{\partial d_{jk}} 
  \frac{\partial d_{jk}}{\partial \mathbf{y_i}}$$

where $d_{jk}$ is the distance between point $j$ and $k$ and we have a double
sum over all pairs of points. These derivatives are all zero unless either 
$j = i$ or $k = i$, so we can simplify to:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_k^N \frac{\partial C}{\partial d_{ik}} 
    \frac{\partial d_{ik}}{\partial \mathbf{y_i}}
+
  \sum_j^N \frac{\partial C}{\partial d_{ji}} 
    \frac{\partial d_{ji}}{\partial \mathbf{y_i}}$$

We can then relabel $k$ to $j$ and move both terms inside the same sum:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N \frac{\partial C}{\partial d_{ij}} 
    \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
+
    \frac{\partial C}{\partial d_{ji}}
    \frac{\partial d_{ji}}{\partial \mathbf{y_i}}$$

Because distances are symmetric, $d_{ij} = d_{ji}$, we can simplify to:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N \left(
    \frac{\partial C}{\partial d_{ij}} +
    \frac{\partial C}{\partial d_{ji}}
   \right) 
   \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
    $$

What we can't do is treat $\frac{\partial C}{\partial d_{ij}}$ and 
$\frac{\partial C}{\partial d_{ji}}$ as equivalent (any of the other variables
$C$ and $d$ are coupled through might be asymmetric).

While there may be some exotic situations where the output distances should be
non-Euclidean (a literary analysis of HP Lovecraft perhaps), I'm not aware of
any publications that do this. You can safely assume that $d_{ij}$ represent
Euclidean distances. In an $K$-dimensional output space, the 
distance between point $\mathbf{y_i}$ and point $\mathbf{y_j}$ is:

$$d_{ij} = \left[\sum_l^K\left (y_{il} - y_{jl} \right )^2\right ]^{1/2}$$

and the derivative can be written as:

$$\frac{\partial d_{ij}}{\partial \mathbf{y_i}} = 
\frac{1}{d_{ij}}\left(\mathbf{y_i} - \mathbf{y_j}\right)$$

### Transformed distance, $f_{ij}$

Now we need an expression for $\frac{\partial C}{\partial d_{ij}}$:

$$\frac{\partial C}{\partial d_{ij}} = 
\sum_k^N \sum_l^N \frac{\partial C}{\partial f_{kl}} 
  \frac{\partial f_{kl}}{\partial d_{ij}}$$

which is only non-zero when $i = k$ and $j = l$, so:

$$\frac{\partial C}{\partial d_{ij}} = 
\frac{\partial C}{\partial f_{ij}} 
\frac{\partial f_{ij}}{\partial d_{ij}}$$

What is this $f_{ij}$? It's a transformation of the output distance that will
then be used as input into the similarity kernel. Most authors do indeed include
this function as part of the similarity kernel itself, or even jump straight
to defining the probabilities, but I prefer to split things up more finely,
because I find that this makes dealing with derivatives of different similarity 
kernels easier. Indulge me.

$f_{ij}$ is an increasing function of the distance between points $i$ and $j$
and is invariably merely the square of the distance. While we could include 
other parameters like a "bandwidth" or "precision" that reflects the local 
density of points at $i$, it's better to include that in the similarity kernel.

Allow me to insult your intelligence by writing out the function and derivative 
for the sake of completeness:

$$f_{ij} = d_{ij}^{2}$$

$$\frac{\partial f_{ij}}{\partial d_{ij}} = 2d_{ij}$$

This may look trivial, but it combines very well with the derivative of the 
Euclidean distance we defined in the previous section:

$$
\frac{\partial f_{ij}}{\partial d_{ij}}
\frac{\partial d_{ij}}{\partial \mathbf{y_i}} 
= 
2d_{ij}
\frac{1}{d_{ij}}\left(\mathbf{y_i} - \mathbf{y_j}\right)
=
2\left(\mathbf{y_i} - \mathbf{y_j}\right)
$$

This will come in handy when we want to simplify the full expression of the
gradient later.

### Similarity weight, $w_{ij}$

Next, $\frac{\partial C}{\partial f_{ij}}$ can be written as:

$$\frac{\partial C}{\partial f_{ij}} = 
\sum_k^N \sum_l^N \frac{\partial C}{\partial w_{kl}} 
  \frac{\partial w_{kl}}{\partial f_{ij}}$$
  
once again, this is only non-zero when $i = k$ and $j = l$:

$$\frac{\partial C}{\partial f_{ij}} = 
\frac{\partial C}{\partial w_{ij}} 
\frac{\partial w_{ij}}{\partial f_{ij}}$$

Various functional forms have been used for the weighting function (or 
similarity kernel; I use either term as the mood takes me). We'll get into
specifics later.

### Probability, $q_{ij}$

So far, so good. Those unpleasant looking double sums are just melting away.
Alas, the good times cannot last forever and now we're going to have to do a bit
more work. Using the chain rule on 
$\frac{\partial C}{\partial w_{ij}}$, we get:

$$\frac{\partial C}{\partial w_{ij}} = 
\sum_k^N \sum_l^N \frac{\partial C}{\partial q_{kl}} 
  \frac{\partial q_{kl}}{\partial w_{ij}}$$

This is the equation that relates the weights to the probabilities. The 
probabilities sum to one, so a change to a weight $w_{ij}$ will affect all the
probabilities, not just $q_{ij}$. Therefore, we should see non-zero derivatives 
for some $q_{kl}$ other than when $i = k$ and $j = l$. 

The probabilities are defined by normalizing the weights so they sum to one. As
discussed above, there are two ways to define the probabilities. The point-wise
normalization gives:

$$q_{ij} = \frac{w_{ij}}{\sum_k^N w_{ik}} = \frac{w_{ij}}{S_i}$$

where $S_i$ is the sum of all the weights associated with point $i$, which 
reduces a bit of notational clutter. The pair-wise normalization is:

$$q_{ij} = \frac{w_{ij}}{\sum_k^N \sum_l^N w_{kl}} = \frac{w_{ij}}{S}$$

$S$ is the sum of all the weights involving all pairs, so there's no need for a
subscript. As you can see, the functional form of the two different 
normalization schemes is very similar, so I'll just use the pair-wise form
from now on.

It's important to realize that any particular weight, $w_{ij}$, appears in both 
the expression for its equivalent probability, $q_{ij}$ (where it appears in 
the numerator and denonimator) _and_ in the expression for all the other 
probabilities, $q_{kl}$, where $i \neq k$ and $j \neq l$. In the latter case, it 
appears only in the denominator, but this is what leads to the non-zero 
derivatives.

Thus, we have two forms of the derivative to consider:
$$\frac{\partial q_{ij}}{\partial w_{ij}} = \frac{S - w_{ij}}{S^2} = 
  \frac{1}{S} - \frac{q_{ij}}{S}$$
and:
$$\frac{\partial q_{kl}}{\partial w_{ij}} = 
  -\frac{w_{kl}}{S^2} = 
  -\frac{q_{kl}}{S}$$

They're nearly the same expression, there's just one extra $\frac{1}{S}$ term 
to consider when $i=k$ and $j=l$.

Inserting these expressions into the one we had for the chain rule applied to
$\frac{\partial C}{\partial w_{ij}}$, we get:

$$
\frac{\partial C}{\partial w_{ij}} = 
\sum_k^N \sum_l^N \frac{\partial C}{\partial q_{kl}} 
  \frac{\partial q_{kl}}{\partial w_{ij}} =
\left[\frac{\partial C}{\partial q_{ij}}\frac{1}{S} +
\sum_k^N \sum_l^N \frac{\partial C}{\partial q_{kl}} \left(- \frac{q_{kl}}{S}\right)
\right]
 \frac{\partial q_{kl}}{\partial w_{ij}}
$$

Extract a $\frac{1}{S}$ factor and we're left with:

$$\frac{\partial C}{\partial w_{ij}} = 
\frac{1}{S} 
  \left[ 
    \frac{\partial C}{\partial q_{ij}} -
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl}
  \right]
 \frac{\partial q_{kl}}{\partial w_{ij}}
$$

I'll admit, that doesn't look great, but we're over the worst.

### Cost function, $C$

Nearly there! Finally, we need to... wait, no, that's it. All that's left is 
an expression for the cost function in terms of the probabilities. And that's 
exactly how the divergences are normally expressed. No chain rule here, 
we can just write $\frac{\partial C}{\partial q_{ij}}$.

## Putting it all together

By substituting in the various expressions, starting with 
$\frac{\partial C}{\partial {d_{ij}}}$ and then recursively replacing any
expressions until we hit $\frac{\partial C}{\partial q_{ij}}$, we can now 
write:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N \left(
  \frac{1}{S} 
  \left[ 
    \frac{\partial C}{\partial q_{ij}} -
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl}
  \right]
    \frac{\partial w_{ij}}{\partial f_{ij}}
    \frac{\partial f_{ij}}{\partial d_{ij}}
  +\frac{1}{S} 
  \left[ 
    \frac{\partial C}{\partial q_{ji}} -
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl}
  \right]
    \frac{\partial w_{ji}}{\partial f_{ji}}
    \frac{\partial f_{ji}}{\partial d_{ji}}    
   \right) 
   \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
    $$

Well, that looks terrifying. Let's tidy up a bit. First, let's use the fact that
the definition of $f_{ij}$ is symmetric and therefore 
$\frac{\partial f_{ij}}{\partial d_{ij}} = 
 \frac{\partial f_{ji}}{\partial d_{ji}}$ to pull that part of the equation out
 of those parentheses:
 
$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N \left(
  \frac{1}{S} 
  \left[ 
    \frac{\partial C}{\partial q_{ij}} -
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl}
  \right]
    \frac{\partial w_{ij}}{\partial f_{ij}}
  +\frac{1}{S} 
  \left[ 
    \frac{\partial C}{\partial q_{ji}} -
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl}
  \right]
    \frac{\partial w_{ji}}{\partial f_{ji}}
  \right)
   \frac{\partial f_{ij}}{\partial d_{ij}}
   \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
    $$

That leaves the two functional forms that get varied the most, 
$\frac{\partial C}{\partial q_{ij}}$ (derivative of the cost function) and 
$\frac{\partial w_{ij}}{\partial f_{ij}}$ (derivative of the similarity 
function), together. There are some common  choices of cost and similarity 
function that would let us simplify these further, but for now we'll leave them 
in their still mildly intimidating forms. Instead, we'll just hide their full 
"glory" by defining:

$$k_{ij} =
  \frac{1}{S} 
  \left[ 
    \frac{\partial C}{\partial q_{ij}} -
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl}
  \right]
    \frac{\partial w_{ij}}{\partial f_{ij}}
$$

And now we can say:
$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N 
  \left(
    k_{ij} + k_{ji}
  \right)
   \frac{\partial f_{ij}}{\partial d_{ij}}
   \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
$$

Further, we can put to good use the expression we came up with for 
$\frac{\partial f_{ij}}{\partial d_{ij}}
\frac{\partial d_{ij}}{\partial \mathbf{y_i}}$ earlier (the one that assumes 
the use of squared Euclidean distances) to get a probability-based embedding
master equation:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j^N 
  \left(
    k_{ij} + k_{ji}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

where

$$k_{ij} =
  \frac{1}{S} 
  \left[ 
    \frac{\partial C}{\partial q_{ij}} -
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl}
  \right]
    \frac{\partial w_{ij}}{\partial f_{ij}}
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

### Some Cost Functions and their Derivatives

#### Kullback-Leibler Divergence

Used by the ASNE, SSNE, t-SNE and variants:

$$C_{ij} = D_{KL}(p_{ij}||q_{ij}) = p_{ij}\ln\left(\frac{p_{ij}}{q_{ij}}\right)$$
$$\frac{\partial C}{\partial q_{ij}} = - \frac{p_{ij}}{q_{ij}}$$

#### "Reverse" Kullback-Leibler Divergence

[NeRV](http://www.jmlr.org/papers/v11/venna10a.html) symmetrizes the KL 
divergence by also considering the cost when $q_{ij}$ is the "reference" 
probability distribution:

$$C_{ij} = D_{KL}(q_{ij}||p_{ij}) = q_{ij}\ln\left(\frac{q_{ij}}{p_{ij}}\right)$$
$$\frac{\partial C}{\partial q_{ij}} = \ln\left(\frac{p_{ij}}{q_{ij}}\right) + 1$$

#### Jensen-Shannon Divergence

The Jensen-Shannon Divergence, as defined in 
[JSE](https://dx.doi.org/10.1016/j.neucom.2012.12.036) is:

$$C_{ij} = D_{JS}(p_{ij}||q_{ij}) = 
\frac{1}{1-\kappa}D_{KL}(p_{ij}||z_{ij}) + 
\frac{1}{\kappa}D_{KL}(q_{ij}||z_{ij})$$
where $Z$ is a mixture of P and Q:
$$Z = \kappa P + \left(1-\kappa \right)Q$$

Rather than invoke the chain rule to couple $Z$ to $Q$, I'm just going to write
out the derivatives for the two parts of the JS divergence with respect to $Q$
(ignoring the $\kappa$ weighting for now):

$$\frac{\partial D_{KL}(p_{ij}||z_{ij})}{\partial q_{ij}} = 
- \left(1 - \kappa \right) \frac{p_{ij}}{z_{ij}}$$
$$\frac{\partial D_{KL}(q_{ij}||z_{ij})}{\partial q_{ij}} = 
\kappa \left(\frac{p_{ij}}{z_{ij}}\right) 
- \ln\left(\frac{q_{ij}}{z_{ij}}\right)$$

Once you add these derivatives together, multiplying by the $\kappa$ values in
the cost function, terms cancel to leave a surprisingly simple derivative:

$$\frac{\partial C}{\partial q_{ij}} = 
\frac{\partial D_{JS}(p_{ij}||q_{ij})}{\partial q_{ij}} =
-\frac{1}{\kappa}\ln\left(\frac{q_{ij}}{z_{ij}}\right)$$

### Some Similarity Kernels and their Derivatives

#### Exponential/Gaussian Kernel

The weighting function most commonly encountered is a gaussian kernel, although
because I've separated the distance transformation from the weighting, it's 
actually exponential with respect to the transformed distances:

$$w_{ij} = \exp\left(-\beta_{i} f_{ij}\right)$$

Also, I've thrown in the exponential decay factor, $\beta_{i}$ which is used in
the [Input Initialization](input-initialization.html) for the input kernel
only, as part of the perplexity-based calibration (aka entropic affinities). 
As discussed further in that section, the output kernel function normally
just sets all the $beta$ values to one, so it effectively disappears from
the output kernel gradient. However, so methods do make use of an output
$\beta$, so here is the general gradient:

$$\frac{\partial w_{ij}}{\partial f_{ij}} 
= -\beta_{i} \exp\left(-\beta_{i} f_{ij}\right)
= -\beta_{i} w_{ij}
$$

#### t-Distribution Kernel

As used in the output kernel of t-SNE. Also referred to as the Cauchy 
distribution sometimes.

$$w_{ij} = \frac{1}{1 + f_{ij}}$$
$$\frac{\partial w_{ij}}{\partial f_{ij}} 
= -\frac{1}{\left(1 + f_{ij}\right)^2}
= -w_{ij}^2
$$

#### Heavy-tailed Kernel

A generalization of the exponential and t-distributed kernel, and described in
the 
[HSSNE paper](https://papers.nips.cc/paper/3770-heavy-tailed-symmetric-stochastic-neighbor-embedding).

$$w_{ij} = \frac{1}{\left(\alpha \beta_{i} f_{ij} + 1\right)^{\frac{1}{\alpha}}}$$
$$\frac{\partial w_{ij}}{\partial f_{ij}} 
= - \frac{\beta_{i}}{\left(\alpha \beta_{i} f_{ij} + 1\right)^{\frac{\alpha+1}{\alpha}}}
= -\beta_{i} w_{ij} ^ \left(\alpha + 1\right)
$$

The degree of heavy-tailedness is controlled by $\alpha$: when $\alpha = 1$, the
kernel behaves like the t-distributed kernel. As it approaches 0, it approaches
the exponential kernel. $\alpha > 1$ allows output distances to stretch even
more than t-SNE. I am unaware of any research that looks at seeing if there is
a way to find an optimal value of $\alpha$ for a given dataset. 

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

### Deriving the SNE and t-SNE Gradient

With a master equation and an expression for the derivative of the cost function
and kernel function, we have all we need to mix and match various costs and
kernels to our heart's content. But it would be nice to see the familiar SNE
and t-SNE gradient fall out of that mixture. Normally, due to reasons of space,
this doesn't get shown and we're left with something like "and then a miracle 
of algebra occurs!" before being shown the final gradient.

Plenty of space on this web page, though. Let's do it. This is the master
equation again:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j^N 
  \left(
    k_{ij} + k_{ji}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

where

$$k_{ij} =
  \frac{1}{S} 
  \left[ 
    \frac{\partial C}{\partial q_{ij}} -
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl}
  \right]
    \frac{\partial w_{ij}}{\partial f_{ij}}
$$

Time to simplify the expression for $k_{ij}$. Both SNE and t-SNE use the 
Kullback-Leibler Divergence, which as noted above, has the following
gradient:

$$\frac{\partial C}{\partial q_{ij}} = - \frac{p_{ij}}{q_{ij}}$$

$k_{ij}$ therefore becomes:

$$k_{ij} =
\frac{1}{S}
  \left[ 
  -\frac{p_{ij}}{q_{ij}} -
    \sum_k^N \sum_l^N 
      -\frac{p_{kl}}{q_{kl}} q_{kl}
  \right]
  \frac{\partial w_{ij}}{\partial f_{ij}} =
\frac{1}{S} 
\left[
  -\frac{p_{ij}}{q_{ij}} +
  \sum_k^N \sum_l^N -p_{kl}
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
\frac{\partial w_{ij}}{\partial f_{ij}} = w_{ij}^n
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
\left(
  -\frac{p_{ij}}{q_{ij}} +
  1
\right)
$$

using the fact that $\frac{w_{ij}}{S} = q_{ij}$. Now, we can move $-q_{ij}$ 
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
  \sum_j^N 
  \left(
    k_{ij} + k_{ji}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

For SNE, $w_{ij}^{n-1} = 1$ because $n = 1$ and we get:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j^N 
  \left(
    p_{ij} - q_{ij} + p_{ji} - q_{ji}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

In asymmetric SNE, the probability matrices are not symmetric due to the 
point-wise normalization, so that's the final gradient. Feel free to write
$p_{ij}$ as $p_{j|i}$ and you are done.

For Symmetric SNE, both the $P$ and $Q$ matrices are symmetric, 
so $p_{ij} = p_{ji}$, and $q_{ij} = q_{ji}$, leading to:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j^N 
  \left(
    p_{ij} - q_{ij} + p_{ji} - q_{ji}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right) =
  2
  \sum_j^N 
  \left(
    2 p_{ij} - 2 q_{ij}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right) =
  4
  \sum_j^N 
  \left(
    p_{ij} - q_{ij}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

Also familiar. For t-SNE, we get:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2
  \sum_j^N 
  \left(
    w_{ij}\left(p_{ij} - q_{ij}\right) + w_{ji}\left(p_{ji} - q_{ji}\right)
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

but the t-distributed kernel also creates a symmetric $W$ matrix so we can 
still simplify to:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  4
  \sum_j^N 
  w_{ij}
  \left(
    p_{ij} - q_{ij}
  \right)
  \left(
   \mathbf{y_i - y_j}
  \right)
$$

I think we all deserve a long lie down now.

## Distance-based embedding

For completeness, here's the derivation of gradients for distance-based
embeddings, such as metric MDS and Sammon Maps. The gradients for these
are well known, so if we can't derive them, we're in big trouble.

An embedding where the cost function is written in terms of the distances
is delightfully straightforward compared to what we just went through. The
chain of variable dependencies is $C \rightarrow d \rightarrow \mathbf{y}$.

We wrote all this out before, so we can just straight to:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N \left(
    \frac{\partial C}{\partial d_{ij}} +
    \frac{\partial C}{\partial d_{ji}}
   \right) 
   \frac{\partial d_{ij}}{\partial \mathbf{y_i}}
$$

and let's go ahead and just assume Euclidean distances:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  \sum_j^N \left(
    \frac{\partial C}{\partial d_{ij}} +
    \frac{\partial C}{\partial d_{ji}}
   \right) 
   \frac{1}{d_{ij}}\left(\mathbf{y_i - y_j}\right)
$$

Unlike the probability-based embedding, this time we do have a direct dependency 
of $C$ on $d$, and the symmetry of distances means that
$\frac{\partial C}{\partial d_{ij}} = \frac{\partial C}{\partial d_{ji}}$, so:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  2\sum_j^N
    \frac{\partial C}{\partial d_{ij}} 
   \frac{1}{d_{ij}}\left(\mathbf{y_i - y_j}\right)
$$

which makes our master equation for distance-based embeddings:

$$
\frac{\partial C}{\partial \mathbf{y_i}} = 
  2\sum_j^N k_{ij} \left(\mathbf{y_i - y_j}\right)
$$
where
$$k_{ij} = \frac{\partial C}{\partial d_{ij}}\frac{1}{d_{ij}}$$
  
### Distance-based Costs

Let's run over some distance-based cost functions, plug them into the  and see what comes.

#### Metric MDS

For a standard metric MDS, the cost for a pair is just the square loss between 
the input distances $r_{ij}$ and the output distances $d_{ij}$:

$$C_{ij} = \left(r_{ij} - d_{ij}\right)^2$$
$$\frac{\partial C}{\partial d_{ij}} = -2\left(r_{ij} - d_{ij}\right)$$

Plugging this into our expression for $k_{ij}$, we get:

$$k_{ij} = -2\frac{\left(r_{ij} - d_{ij}\right)}{d_{ij}}$$

And then the gradient is:
$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  -4\sum_j^N \frac{\left(r_{ij} - d_{ij}\right)}{d_{ij}}
$$

This may not be quite what you're expecting, because most distance-based
cost functions take advantage of the symmetry of the distance matrix and 
generally only sum where the indices are $i < j$. In order to be consistent
with non-symmetric cost functions, I'm considering the entire matrix.

#### SSTRESS

The SSTRESS criterion is

$$C_{ij} = \left(r_{ij}^2 - d_{ij}^2\right)^2$$
$$\frac{\partial C}{\partial d_{ij}} = -4d_{ij}\left(r_{ij}^2 - d_{ij}^2\right)$$

The stiffness is therefore:
$$k_{ij} = -4\left(r_{ij}^2 - d_{ij}^2\right)$$

and the gradient is:
$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  =
  -8\sum_j^N \left(r_{ij}^2 - d_{ij}^2\right) \left(\mathbf{y_i - y_j}\right)
$$

There is some interesting discussion of artefactual structure that can result 
from the use of the SSTRESS criterion in a paper by 
[Hughes and Lowe](https://papers.nips.cc/paper/2239-artefactual-structure-from-least-squares-multidimensional-scaling).

#### Sammon Mapping

Sammon Mapping is very similar to metric MDS, except that it tries to put
more weight on preserving short distances (and hence local structure), by
weighting the cost function by $\frac{1}{r_{ij}}$:

$$C_{ij} = \frac{\left(r_{ij} - d_{ij}\right)^2}{r_{ij}}$$
$$\frac{\partial C}{\partial d_{ij}} = 
  -2\frac{\left(r_{ij} - d_{ij}\right)}{r_{ij}}$$

Therefore:

$$k_{ij} = -2\frac{\left(r_{ij} - d_{ij}\right)}{r_{ij}d_{ij}}$$

and the gradient is:

$$\frac{\partial C}{\partial \mathbf{y_i}} = 
  -4\sum_j^N \frac{\left(r_{ij} - d_{ij}\right)}{r_{ij}d_{ij}}
$$

Once again, as normally written (e.g. in 
[Sammon's original paper](https://dx.doi.org/10.1109/T-C.1969.222678)), the 
constant term is twice too big (i.e. you'll normally see a $-2$, not a $-4$), 
because you wouldn't normally bother to sum over $i > j$. Also, the Sammon 
error includes a normalization term (the sum of all $\frac{1}{r_{ij}}$), but 
that's a constant for any configuration, so irrelevant for the purposes of 
deriving a gradient.

If you've been able to follow along with this, you surely are now practiced 
enough to derive gradients for more experimental embedding methods. The
next section gives some examples.

Next: [Experimental Gradients](experimental-gradients.html) Up: [Index](index.html)
