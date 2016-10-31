---
title: "Embedding Methods"
output: html_document
---
Previous: [Optimization](optimization.html). Next: [Reporting](reporting.html). Up: [Index](index.html).

### `method`

By default, `sneer` carries out t-SNE. But other optimization methods are
available, which you can access via the `method` argument. The currently
available methods are:

#### Principal Component Analysis (PCA)

This is neither a distance-based embedding or a probability-based embedding. 
It just carries out a Principal Component Analysis of the data set and uses the 
first two principal components as the final coordinates. It's exactly the
same as when you use the PCA option for 
[Output Initialization](output-initialization.html) but do no further 
optimization. As a result, this method ignores the `init` option.

```R
iris_pca <- sneer(iris, method = "pca")
```

#### Metric Multi Dimensional Scaling (MDS)

There are a whole lot of techniques out there under the rubric 'MDS'. Just to
be totally clear, the version of MDS that is used here is one that arises 
naturally from the idea of minimizing the difference between the input
and output distances. The cost function is simply the square loss between 
the input and output distance matrices. There is no re-scaling or any other
funny business.

```R
iris_mmds <- sneer(iris, method = "mmds")
```

I don't exactly commend this as a shining example of an efficient MDS routine.
But it's the very simplest possible distance-based embedding. I recommend 
trying this and/or PCA as a sanity check with new data sets before embarking
on the more exotic methods on offer.

#### Sammon Map

Sammon mapping works like metric MDS but it adds an extra weighting to put more
emphasis on reproducing short distances rather than long distances. In practice
it doesn't produce results that are all that different from metric MDS.

```R
iris_sammon <- sneer(iris, method = "sammon")
```

#### Asymmetric Stochastic Neighbor Embedding (ASNE)

This is the original SNE method as described in the
[SNE paper](https://papers.nips.cc/paper/2276-stochastic-neighbor-embedding),
but it's now referred to as "asymmetric" SNE to differentiate it from other 
variants.

ASNE is the prototypical probability-based embedding method. It uses the 
Kullback-Leibler divergence as a cost function. A big difference between
it and t-SNE, is that the output kernel is a gaussian, rather than the
t-distribution with one degree of freedom used in t-SNE.

```R
iris_asne <- sneer(iris, method = "asne")
```

#### Symmetric Stochastic Neighbor Embedding (SSNE)

The [SSNE paper (PDF)](https://www.cs.toronto.edu/~amnih/papers/sne_am.pdf) 
differentiates itself from ASNE by changing how the normalization procedure
works: it does it using the entire weight matrix, rather than per-row of the
matrix. See the [Gradient](gradients.html) theory page for the difference. 

In general, I find "symmetric" methods that use the SSNE version of 
normalization versus the ASNE version tend to optimize a little more easily
and show detectable, albeit often very minor, differences in the final 
configuration. Not everyone agrees, though. For instance, the authors
of [JSE](http://dx.doi.org/10.1016/j.neucom.2012.12.036) noted "no significant
effect" of the normalization procedure on the results they presented.

```R
iris_ssne <- sneer(iris, method = "ssne")
```

#### t-Distributed SNE (t-SNE)

As mentioned, t-SNE is the default method. But to be really sure you can
provide `"tsne"` as the argument:

```R
iris_tsne <- sneer(iris, method = "tsne")
```

#### Weighted Symmetric SNE (ws-SNE)

This method scales each weight by its "importance", which is related to the
input probability. The higher the probability, the more important it is.
See the [ws-SNE paper (PDF)](http://jmlr.org/proceedings/papers/v32/yange14.pdf)
for the details. Also, note that this is a modification of t-SNE, rather than
SSNE.

```R
iris_wssne <- sneer(iris, method = "wssne")
```

#### Heavy-tailed Symmetric Stochastic Neighbor Embedding (HSSNE)

[HSSNE](http://papers.nips.cc/paper/3770-heavy-tailed-symmetric-stochastic-neighbor-embedding)
generalizes SSNE and t-SNE, by introducing a tail-heaviness parameter, `alpha`,
which takes a value between `0` (behaves like `method = "ssne"`),
and `1` (in which case you will get the behavior of `method = "tsne"`). 

```R
iris_hssne <- sneer(iris, method = "hssne", alpha = 0) # SSNE
iris_hssne <- sneer(iris, method = "hssne", alpha = 0.5) # default HSSNE
iris_hssne <- sneer(iris, method = "hssne", alpha = 1) # t-SNE
```

If t-SNE tends to spend a lot of the optimization slowly separating clusters,
I suggest trying HSSNE with a value of `alpha` below `1` - you may find it 
produces a better, more compact final result.

#### Neighbor Retrieval Visualizer (NeRV)

As implemented in `sneer`, NeRV is a generalization of ASNE, that uses 
a symmetrized version of the Kullback-Leibler divergence.

```R
iris_nerv <- sneer(iris, method = "nerv")
```

The degree of symmetrization of the KL divergence can be controlled with the 
`lambda` parameter which can take a value between `0` and `1` and defaults
to `0.5`. When it's set to 1, NeRV should perform just like ASNE. At zero, it 
effectively uses a "reverse" Kullback-Leibler divergence, which has the effect 
of strongly penalizing embedded points which get too close together.

```R
iris_nerv <- sneer(iris, method = "nerv", lambda = 0.5) # default NeRV
iris_nerv <- sneer(iris, method = "nerv", lambda = 1) # ASNE
iris_nerv <- sneer(iris, method = "nerv", lambda = 0) # "reverse" ASNE
```

See the [NeRV](http://www.jmlr.org/papers/v11/venna10a.html) paper for more
details. In that publication, they suggest transferring the input kernel
bandwidths to the output kernel. To do this, set the `prec_scale` parameter 
with an argument of `"t"`

```R
iris_nerv <- sneer(iris, method = "nerv", prec_scale = "t") # original NeRV
```

Later publications (see, for example, the 
[ws-SNE paper (PDF)](http://jmlr.org/proceedings/papers/v32/yange14.pdf)) don't
mention this difference and I recommend not setting `prec_scale`. See the
[Input Initialization](input-initialization.html) section for more on 
`prec_scale`.

#### Jensen-Shannon Embedding (JSE)

[JSE](http://dx.doi.org/10.1016/j.neucom.2012.12.036) is a technique that is 
similar to NeRV in that it uses a symmetrized version of a divergence as 
a cost function. In the case of JSE, however, it used the Jensen-Shannon
divergence, rather than the Kullback-Leibler divergence. To try it, set 
`method = "jse"`:

```R
iris_jse <- sneer(iris, method = "jse")
```

Like NeRV, JSE has a parameter to control the degree of symmetrization of the
cost function. And like NeRV, it takes a value between `0` and `1`, with its
default being set to `0.5`. And like NeRV, at one extreme, you get an embedding
that's the same as ASNE. Except for JSE, the parameter is called `kappa`, and
you get ASNE-like behavior when it's set to `0`:

```R
iris_jse <- sneer(iris, method = "jse", kappa = 0.5) # default JSE
iris_jse <- sneer(iris, method = "jse", kappa = 1) # "reverse" ASNE
iris_jse <- sneer(iris, method = "jse", kappa = 0) # ASNE
```

Due to numerical issues with how the JSE cost function and gradient is 
calculated, don't expect to get results exactly like ASNE/NeRV when setting
`kappa` to its minimum and maximum, but it gets close.

If you find it confusing that there are two separate parameters with Greek 
letter names, that do nearly the same thing, but apply to two different
embedding methods, you're not alone, but I decided to try and stick with
the nomenclature used in the original publications wherever possible, to make
comparison with literature results easier.

#### Console log

The details of what appears during optimization is covered in the 
[Reporting](reporting.html) section. Just be aware that the choice of `method`
changes the cost function that is being optimized, which in turn will affect
the meaning of the `cost` and `norm` values that are output. Don't attempt to 
compare these values between different methods.

Previous: [Optimization](optimization.html). Next: [Reporting](reporting.html). Up: [Index](index.html).
