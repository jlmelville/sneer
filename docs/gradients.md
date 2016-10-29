---
title: "Deriving Embedding Gradients"
output: html_document
---

There's no R code in this document. Instead, here's my attempt to derive
the gradient for various embedding methods. There are a few papers which attempt
something like this, the clearest in my opinion being the treatment by 
Lee et al. The version I give here is very similar, but with a bit less 
notation.

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

I assume you are familiar with the basics of the approach of SNE and related 
methods. I'll use the following notation:

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
approach. The resulting matrix $Q$ contains a single probability distribution, 
i.e. the grand sum of the matrix is one. Using this normalization, it's still
true that, in general, $q_{ij} \neq q_{ji}$, but when creating the input 
probabilities $p_{ij}$, $p_{ij}$ and $p_{ji}$ are averaged so that they are 
equal to each other. In the case of the output weights, the function that 
generates them  always produces symmetric weights, so that $w_{ij} = w_{ji}$ 
which naturally leads to $q_{ij} = q_{ji}$, so the resulting matrix is 
symmetric without having to do any extra work.

This pair-wise scheme is used in what is called Symmetric SNE and t-distributed
SNE.

Obviously these two schemes are very similar to each other, but it's easy to
get confused when looking at how different embedding methods are implemented.
As to whether it makes much of a practical difference, Lee and co-workers say 
that it has "no significant effect" on JSE, whereas van der Maaten and Hinton 
note that SSNE sometimes produced results that were "a little better" than ASNE.
Not a ringing endorsement either way, but in my experiments with sneer, the
symmetrized (pair-wise) normalization seems to produce better results.

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

Inserting these expressions into the one we had for the chain rule applied to
$\frac{\partial C}{\partial w_{ij}}$, we get:

$$\frac{\partial C}{\partial w_{ij}} = 
-\frac{1}{S} 
  \left[ 
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl} + 
      \frac{\partial C}{\partial q_{ij}} 
  \right]
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
    -\frac{1}{S} 
  \left[ 
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl} + 
      \frac{\partial C}{\partial q_{ij}} 
  \right]
    \frac{\partial w_{ij}}{\partial f_{ij}}
    \frac{\partial f_{ij}}{\partial d_{ij}}
    -\frac{1}{S} 
  \left[ 
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl} + 
      \frac{\partial C}{\partial q_{ji}} 
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
    -\frac{1}{S} 
  \left[ 
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl} + 
      \frac{\partial C}{\partial q_{ij}} 
  \right]
    \frac{\partial w_{ij}}{\partial f_{ij}}
    -\frac{1}{S} 
  \left[ 
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl} + 
      \frac{\partial C}{\partial q_{ji}} 
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
-\frac{1}{S} 
  \left[ 
    \sum_k^N \sum_l^N 
      \frac{\partial C}{\partial q_{kl}} q_{kl} + 
      \frac{\partial C}{\partial q_{ij}} 
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
the use of squared Euclidean distances) to get:

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

This is now looking more like the expected "points on springs" interpretation of
the gradient, with the $k_{ij}$ representing the force constant (stiffness) of 
each spring, and $\mathbf{y_i - y_j}$ the displacement.

The above equation is useful because as long as you can define the gradient
of a cost function in terms of $q$ and the gradient of a similarity kernel in
terms of $f$, you can mix and match these terms and get the gradient of the
cost function with respect to the embedded coordinates without too much trouble,
which is all you need to optimize the coordinates with a standard gradient
descent algorithm.

## Other Cost Functions

### Distance-based embedding

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
  =
  2\sum_j^N k_{ij} \left(\mathbf{y_i - y_j}\right)
$$

where

$$k_{ij} = \frac{\partial C}{\partial d_{ij}}\frac{1}{d_{ij}}$$

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

### SSTRESS

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

### Sammon Mapping

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

Once again, as normally written (e.g. in Sammon's original paper), the constant
term is twice too big (i.e. you'll normally see a $-2$, not a $-4$), because you
wouldn't normally bother to sum over $i > j$. Also, the Sammon error includes
a normalization term (the sum of all $\frac{1}{r_{ij}}$), but that's a constant
for any configuration, so irrelevant for the purposes of deriving a gradient.

## Generalized Divergences

Divergences are normally applied to probabilities. But they have been 
generalized to work with other values. Some of this is discussed by Yang and 
co-workers (in Optimization Equivalence of Divergences Improves Neighbor
Embedding), using definitions by Cichocki and co-workers.

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

$$k_{ij} = \frac{\partial C}{\partial w_{ij}}
\frac{\partial w_{ij}}{\partial f_{ij}}$$

The generalized divergences are defined in terms of the output weights, 
$w_{ij}$ and the input weights, which I've not had to come up with a symbol 
before now. Let's call them $v_{ij}$. The generalized Kullback Leibler 
divergence (also known as the I-divergence, and apparently used in non-negative
matrix factorization) and its derivative with respect to the output weights is:

$$C_{ij} = v_{ij}\ln\left(\frac{v_{ij}}{w_{ij}}\right) - v_{ij} + w_{ij}$$
$$\frac{\partial C}{\partial w_{ij}} = 1 - \frac{v_{ij}}{w_{ij}}$$

I'm not aware of any embedding algorithms that define their cost function in 
this way, although the Yang paper shows that elastic embedding can be considered
a variant of this.

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

## Distance-based gradient with transformed distances

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




