---
title: "Embedding Methods"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---
Previous: [Optimization](optimization.html). Next: [Reporting](reporting.html). Up: [Index](index.html).

## `method`

By default, `sneer` carries out t-SNE. But other optimization methods are
available, which you can access via the `method` argument. The currently
available methods are:

### Principal Component Analysis (PCA)

This is neither a distance-based embedding or a probability-based embedding. 
It just carries out a Principal Component Analysis of the data set and uses the 
first two principal components as the final coordinates. It's exactly the
same as when you use the PCA option for 
[Output Initialization](output-initialization.html) but do no further 
optimization. As a result, this method ignores the `init` option.

```R
iris_pca <- sneer(iris, method = "pca")
```

### Metric Multi Dimensional Scaling (MDS)

There are a whole lot of techniques out there under the rubric "MDS". Just to
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

### Sammon Map

Sammon mapping works like metric MDS but it adds an extra weighting to put more
emphasis on reproducing short distances rather than long distances. In practice
it doesn't produce results that are all that different from metric MDS.

```R
iris_sammon <- sneer(iris, method = "sammon")
```

### Asymmetric Stochastic Neighbor Embedding (ASNE)

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

### Symmetric Stochastic Neighbor Embedding (SSNE)

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

### t-Distributed SNE (t-SNE)

As mentioned, t-SNE is the default method. But to be really sure you can
provide `"tsne"` as the argument:

```R
iris_tsne <- sneer(iris, method = "tsne")
```

Compared to SSNE, the heavier tail of the output kernel in t-SNE allows close
neighbors to take up larger distances in the output configuration.

### t-Distributed ASNE (t-ASNE)

This is not a literature method. But if t-SNE is a modified version of SSNE,
does an equivalent version exist for ASNE? In `sneer`, the answer is yes, 
although I'm not aware of any literature that explicitly defines and studies
it, although parts of the literature on NerV and inhomogeneous t-SNE 
(see below) touch on it. 

```R
iris_tasne <- sneer(iris, method = "tasne")
```

### Weighted Symmetric SNE (ws-SNE)

This method scales each weight by its "importance", which is related to the
input probability. The higher the probability, the more important it is.
See the [ws-SNE paper](http://jmlr.org/proceedings/papers/v32/yange14.html)
for the details. Also, note that this is a modification of t-SNE, rather than
SSNE.

```R
iris_wssne <- sneer(iris, method = "wssne")
```

### Heavy-tailed Symmetric Stochastic Neighbor Embedding (HSSNE)

[HSSNE](http://papers.nips.cc/paper/3770-heavy-tailed-symmetric-stochastic-neighbor-embedding)
generalizes SSNE and t-SNE, by introducing a tail-heaviness parameter, `alpha`,
which takes a value between `0` (behaves like `method = "ssne"`),
and `1` (in which case you will get the behavior of `method = "tsne"`).

```R
iris_hssne <- sneer(iris, method = "hssne", alpha = 0) # SSNE
iris_hssne <- sneer(iris, method = "hssne", alpha = 0.5) # default HSSNE
iris_hssne <- sneer(iris, method = "hssne", alpha = 1) # t-SNE
```

Actually, you can set `alpha` to values > 1, for those times when the 
tail-heaviness of t-SNE just isn't heavy enough. It would be interesting to
see a data set where a value of alpha much greater than `1` helped 
significantly.

Conversely, for lower-dimensional data sets, if t-SNE tends to spend a lot of the 
optimization slowly separating clusters, that might mean that t-SNE is just
too heavy. I suggest trying HSSNE with a value of `alpha` below `1` - you may 
find it produces a better, more compact final result.

### Neighbor Retrieval Visualizer (NeRV)

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
precisions directly to the output kernel. To do this, set the `prec_scale` 
parameter with an argument of `"t"`

```R
iris_nerv <- sneer(iris, method = "nerv", prec_scale = "t") # original NeRV
```

Later publications (see, for example, the 
[ws-SNE paper](http://jmlr.org/proceedings/papers/v32/yange14.html)) don't
mention this difference and I recommend not setting `prec_scale`. See the
[Input Initialization](input-initialization.html) section for more on 
`prec_scale`.

### Jensen-Shannon Embedding (JSE)

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

### Inhomogeneous t-SNE (it-SNE)

[Inhomogeneous t-SNE](http://dx.doi.org/10.1007/978-3-319-46675-0_14) is 
reminiscent of HSSNE, in that it defines a parameter that controls the 
amount of "stretching" that's allowed of the output space compared to
the input space. To editorialize for a moment, it's so similar that I am
surprised that a reference to HSSNE does not appear in this paper.

Anyway, two big differences to HSSNE are that the stretching parameter is
now defined per point, and that each of these values are optimized independently
along with the coordinates.

The initial value of the stretching parameters, which are associated with the
degrees of freedom of the t-distribution, can be controlled by setting 
`dof`. `dof = 1` is the t-distribution as used in t-SNE, and making `dof` 
infinite gives a gaussian distribution.

```R
iris_itsne <- sneer(iris, method = "itsne", dof = 1) # initial t-distribution
iris_itsne <- sneer(iris, method = "itsne", dof = 1e3) # initial Gaussian
```

It's probably better to initialize the `dof` parameters to a larger value,
(the authors suggest `dof = 10`, which is the default) to ensure that the 
gradients are initially large for all points. Otherwise you run the risk of
outlying points losing their gradients and not moving.

The authors also suggest not starting the optimization of `dof` immediately.
Their procedure is the usual t-SNE optimization process of starting from
a small random, normal distribution, using early exaggeration, continuing to
optimize just the coordinates, and then adding optimization of `dof`. When
to start optimizing `dof` is controlled by the `kernel_opt_iter` parameter,
which by default is `50` iterations (if you were using early exaggeration, is 
the default iteration number at which that stops). You may not have to wait
as long (or at all) if you are initializing from a non-random configuration.

```R
# Start optimizing dof straight away
iris_itsne <- sneer(iris, method = "itsne", kernel_opt_iter = 0)
```

If you are curious about what the values of `dof` end up as, you can ask for
the extra parameters to be exported:

```R
iris_itsne <- sneer(iris, method = "itsne", ret = c("dyn"))
# Results are in a vector called dof, itself part of the dyn list
summary(iris_itsne$dyn$dof)
```

See the [Exported Data](exported-data.html) for more on exporting.

Because you are mixing both coordinates and the `dof` parameter for optimizing,
I strongly suggest you stick with a standard optimizer (like the default L-BFGS)
when using inhomogeneous t-SNE. Adding these extra parameters does also slow
down convergence.

Finally, a minor point on nomenclature: like JSE and NerV, it-SNE uses a
point-wise, rather than a pair-wise normalization approach. The embedding is 
therefore more closely related to ASNE and t-ASNE than t-SNE, despite the name.
If you don't like the results you see with it-SNE compared with t-SNE, you
may want to compare with `method = "asne"` and `method = "tasne"`.

### Dynamic HSSNE (DHSSNE)

The similarities between HSSNE and it-SNE inspired me to create the obvious
extension to HSSNE: instead of having to choose a fixed version of HSSNE's
`alpha` parameter, optimize it directly. Because the value of `alpha` fluctuates
throughout the optimization, I call this "dynamic" HSSNE.

Two reasons to choose between DHSSNE and it-SNE are that DHSSNE provides a 
dynamic generalization between SSNE and t-SNE, whereas it-SNE generalizes ASNE
and t-ASNE; and DHSSNE only optimizes a single global parameter. This makes it
less flexible, but it might make convergence a bit easier.

You can continue to specify `alpha` to provide an initial value. Like it-SNE,
it makes sense to initialize it to give gaussian-like behavior (so `alpha = 0`,
which makes it like SSNE initially), although maybe because it's using a single 
global parameter, DHSSNE seems less sensitive to initialization issues than 
it-SNE, which needs to optimize multiple parameters.

The `ret` and `kernel_opt_iter` parameters also apply to `dhssne`. 

```R
# Initialize alpha to 0, don't wait to start optimizing, store optimized alpha
iris_dhssne <- sneer(iris, method = "dhssne", alpha = 0,
                     kernel_opt_iter = 0, ret = c("dyn"))
# Optimized alpha:
iris_dhssne$dyn$alpha
```

I made this method up, so you're not going to find it in any literature 
anywhere. But I think it might be useful. There is some extra background
material on [computing the gradient for DHSSNE](dynamic-hssne.html), if you're 
interested in more details.

## `dyn`

If you're interested in the way that the `itsne` and `dhssne` methods optimize 
the kernel parameters, you can apply this to any method that uses either the 
exponential or heavy-tailed kernel, which is pretty much ever method listed here
except `tsne`, `tasne` and `wtsne`, which don't have any free parameters 
associated with their output kernels. 

To "dynamize" a kernel (my own terrible term), supply the `dyn` parameter with a
named list, where the names are the names of the kernel parameters to optimize,
and the values whether they should be static, global or point-wise optimized
(more on what the acceptable values are below):

### Exponential Kernel

For the exponential kernel, the free parameter is `beta`:

```R
iris_dyn <- sneer(iris, method = "asne", dyn = list(beta = "global"))
```

This should work with methods `asne`, `ssne`, `jse` and `nerv`.

### Heavy-tailed Kernel

For the heavy-tailed kernel, you must specify one or both of `alpha` and `beta`:

```R
iris_dyn <- sneer(iris, method = "hssne", 
                   dyn = list(alpha = "static", beta = "point"))
```

This will work with `hssne`.

The acceptable values of the list and their meanings are:

* `static` - don't optimize this parameter. For the exponential kernel, because
there's only one parameter, this is pointless. This is useful for the 
heavy-tailed kernel though, if you want to only optimize the heavy-tailedness
parameter `alpha`, and not touch the precisions, `beta`.
* `global` - optimize a single global parameter that applies to every point.
* `point` - optimize multiple parameters, with one parameter per point. This 
allows the parameters for the kernel to be optimized for each point individually.

For `dhssne`, where a single value of `alpha` is applied to all points, the
following two commands are equivalent:

```R
iris_dhssne    <- sneer(iris, method = "dhssne")
iris_dyn_hssne <- sneer(iris, method = "hssne", 
                          dyn = list(alpha = "global", beta = "static"))
```

### `itsne` kernel

You can also modify `itsne` in a similar way, although the `itsne` kernel uses
`dof` as a name. It's already dynamic, so this is one place where the use of
`static` in the `dyn` list would have an effect.

Because `itsne` already allows its kernel to be optimized for each point, it's
the equivalent of invoking it like this:

```R
iris_itsne <- sneer(iris, method = "itsne", dyn = list(dof = "point"))
```

but in this case, the `dyn` parameter is redundant. However, you can change
its behavior:

```R
# Optimize one value of dof for all points
iris_itsne <- sneer(iris, method = "itsne", dyn = list(dof = "global"))

# Keep dof fixed to its input value(s) for the entire embedding
iris_itsne <- sneer(iris, method = "itsne", dyn = list(dof = "static"))
```

If you "dynamize" the heavy-tailed, exponential or itsne kernel in this way, you
can get to the final optimized values by passing `ret = c("dyn")` to `sneer` 
just as described for `dhssne` and `itsne` methods above.

## Console log

The details of what appears during optimization is covered in the 
[Reporting](reporting.html) section. Just be aware that the choice of `method`
changes the cost function that is being optimized, which in turn will affect
the meaning of the `cost` and `norm` values that are output. Don't attempt to 
compare these values between different methods.

Previous: [Optimization](optimization.html). Next: [Reporting](reporting.html). Up: [Index](index.html).
