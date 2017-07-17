---
title: "Advanced Embedding Methods"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---
Previous: [Embedding Methods](embedding-methods.html). Next: [Reporting](reporting.html). Up: [Index](index.html).

## The `embedder` function

To gain even more control over the type of embedding that is carried out, you
can also create a custom embedding method with the `embedder` function, the
return value of which you pass as an argument of the `method` parameter of the
`sneer` function, instead of a name of a literature method.

Almost all of the embedding methods available in `sneer` can be created by
specifying the following arguments to `embedder`:

### `cost`

The cost function, which must be one of: 

* `"KL"` The Kullback-Leibler divergence.
* `"reverse-KL"` The "reverse" KL divergence.
* `"NeRV"` The NeRV cost function.
* `"JS"` The Jensen-Shannon divergence, as used in JSE.

### `transform`

A transformation to apply to the distances, before applying a kernel function,
one of:

* `"none"` Don't apply a transformation.
* `"square"` Square the distances.

Most embedding methods square the distances before further processing. Metric
MDS uses the raw distances, however.

### `kernel`

`kernel`: The kernel function used to convert transformed distances to weights, 
which must be one of:

* `"exponential"` The exponential function, as used in ASNE and SSNE.
* `"t-distributed"` The t-distribution, as used in t-SNE.
* `"heavy-tailed"` The heavy-tailed function used in HSSNE.
* `"inhomogeneous"` The function used in inhomogeneous t-SNE.

### `norm`

The normalization to be used when converting the weights to probabilities, which
must be one of:

* `"none"` No normalization, i.e. use the weights in the cost function.
* `"point"` Pointwise normalization, as used in ASNE, NeRV and JSE.
* `"joint"` Pairwise normalization followed by averaging (if required), to 
convert conditional probabilities into joint probabilities, as used in SSNE and
t-SNE.

I've not seen any literature based on SNE where no normalization was applied 
(i.e. setting `norm = "none"`), except for 
[elastic embedding (PDF)](http://faculty.ucmerced.edu/mcarreira-perpinan/papers/icml10.pdf).
See also the paper by 
[Yang, Peltonen and Kaski](http://jmlr.org/proceedings/papers/v32/yange14.html)
. However, distance-based embedding methods such as metric MDS and Sammon 
mapping can be seen as un-normalized methods, where also no kernel is applied.

If you do want to use `norm = "none"`, be aware that modified `cost` functions 
will be used if you choose `"kl"`, `"reverse-KL"` or `"NeRV"`, to account for 
the un-normalized values, using the expression for the Kullback-Leibler
diverence given by 
[Cichocki and co-workers](https://dx.doi.org/10.3390/e13010134). For `"JSE"`, 
cancellation of terms means the normalized and non-normalized expressions are
identical (although the overall gradient expression is more complex).

The `norm` parameter can actually more complex input, but you don't need to
worry about it most of the time. See the section on `joint` vs `pair` 
normalization below for more details.

The above values are case insensitive and can be abbreviated. Here's how to
describe t-SNE, NeRV and JSE using `embedder`:

```R
# t-SNE
tsne_embedder <- embedder(cost = "kl", kernel = "t-dist", norm = "joint")

# NeRV
nerv_embedder <- embedder(cost = "nerv", kernel = "exp", norm = "point")

# JSE
jse_embedder <- embedder(cost = "JS", kernel = "exp", norm = "point")
```

and here's how to create a distance-based embedder, like metric MDS, or metric
MDS using the SSTRESS criterion:

```R
# MDS
mmds_embedder <- embedder(cost = "square", kernel = "none", transform = "none", norm = "none")

# MDS using SSTRESS
smmds_embedder <- embedder(cost = "square", kernel = "none", transform = "square", norm = "none")
```

And why not try an un-normalized version of t-SNE or a normalized version of MDS?
```R
# un-normalized t-SNE
untsne_embedder <- embedder(cost = "kl", kernel = "t-dist", norm = "joint")

# pair-normalized MDS
nmmds_embedder <- embedder(cost = "square", kernel = "none", transform = "none", norm = "pair")
```

Use these methods with `sneer`, via the `method` parameter:

```R
tsne <- embedder(cost = "kl", kernel = "t-dist", norm = "joint")
iris_tsne <- sneer(iris, method = tsne_embedder)
```

For the examples above, there's no reason to use `embedder`, because it's
entirely equivalent to using `method = "tsne"` and so on. But as an example of
a non-literature method, you could create a "symmetrized" version of JSE by 
changing the normalization from `"point"` to `"joint"`:

```R
# SJSE
sjse_embedder <- embedder(cost = "JS", kernel = "exp", norm = "joint")
iris_sjse <- sneer(iris, method = sjse_embedder)
```

### Importance Weighting

The importance weighting technique used in the ws-SNE method can be applied
to any embedding method by passing `importance_weight = TRUE`. We can turn
SJSE into w-SJSE, for example:

```R
# w-SJSE
wsjse_embedder <- embedder(cost = "JS", kernel = "exp", norm = "joint", importance_weight = TRUE)
iris_wsjse <- sneer(iris, method = wsjse_embedder)
```

### `pair` vs `joint` normalization

For most methods, the difference between `pair` and `joint` normalization can
be ignored: just use `joint`. 

`pair` normalization creates the probability $p_{ij}$ from the equivalent
weight $w_{ij}$in the following way:

$$p_{ij} = \frac{w_{ij}}{\sum_k^N \sum_l^N w_{kl}}$$

i.e. we sum over all weights in the data set. `joint` normalization carries
out a further step, by enforcing that the final probabilities are joint by 
averaging:

$$\bar{p}_{ij} = \left(p_{ij} + p_{ji}\right) / 2$$

This second step is important when calibrating the input probabilities because
the input weight matrix is not symmetric. However, for embedding methods which
use pair-wise normalization (SSNE and t-SNE), the output weight matrix *is*
symmetric, so the resulting probabilities are already joint. The extra averaging
step is therefore unnecessary and isn't carried out.

The situation changes if the output weight matrix is not symmetric. There are 
several ways a non-symmetric output weight matrix can occur. For example, the
original NeRV paper suggests using the input kernel parameters for the output
kernel. The it-SNE method allows the kernel parameters to vary per-point. 
However, both these methods use point-wise normalization, so don't have to deal
with this issue.

The brave user of `embedder`, however, may decide they want a pair-wise 
normalization along with a non-symmetric weight matrix. You have the following
options:

* `norm = "pair"`: use pairwise normalization for both the input and
output probabilities. . The averaging step won't be carried out for either the 
input or output probabilities. However, the t-SNE paper warns that this can
cause some points to have very small input probability values. This can lead to
some points being outliers due to their being little contribution to the
gradient.
* `norm = "joint"`: force both input and output probabilities to be 
joint probabilities. This is the standard treatement for input probabilities in
SSNE and t-SNE, but ends up with an extra averaging step per iteration in the 
output probabilities, and means the gradient has a less simple expression for
some embedding methods, which translates to a longer run time.
* `norm = c("joint", "pair")`: force the input probabilities to be 
joint, but don't bother for output probabilities. This is probably the most
pragmatic option, because it gives ensures the input probabilities are 
non-negligible while allowing for the simpler gradient expressions 
(if it exists).
* `normalization - c("pair", "joint")`: the input probabilities are not 
averaged, but the output probabilities are. This is an eccentric choice with
little to recommend it as far as I can see.

What you defintely *shouldn't* do is mix `"point"` normalization with
`"pair"` or `"joint"` normalization. That is an error.

Finally, remember that this is only something you need to think about if you
want to use non-uniform kernel parameters in the output probabilities with
pair-wise normalization.

As an example of where these choices do make a difference, consider the
symmetrized version of JSE we created above. If we use the 
`prec_scale = transfer` option in `sneer`, we will now have a kernel with 
non-uniform parameters and the following embeddings will all give 
(very slightly) different results:

```R
iris_sjse_j <- sneer(iris, prec_scale = "transfer",  method = embedder(cost = "JS", kernel = "exp",
                                                                       norm = "joint"))
iris_sjse_p <- sneer(iris, prec_scale = "transfer",  method = embedder(cost = "JS", kernel = "exp", 
                                                                       norm = "pair"))
iris_sjse_jp <- sneer(iris, prec_scale = "transfer", method = embedder(cost = "JS", kernel = "exp", 
                                                                       norm = c("joint", "pair")))
```
