---
title: "Input Initialization"
output: html_document
---
Previous: [Preprocessing](preprocessing.html). Next: [Output Initialization](output-initialization.html). Up: [Index](index.html).

I consider input [preprocessing](preprocessing.html) options as separate from 
the input initialization options based on the distinction between preprocessing 
as some fairly generic procedures you would probably do a data set before any 
form of machine learning, whereas the input initialization is specific to 
embedding. You can also think of it as a form of calibration.

The degree of input initialization carried out depends on both the type of
embedding and the input data. It can range from doing nothing to some fairly
substantial work, although computationally the time taken is normally dwarfed
by the optimization procedure once the embedding proper kicks off.

### Distance-based embedding

Good news. If you're trying match input to output distances, you have almost
no initialization to do. Probably at most you have to convert the data frame
into a distance matrix. Unless you passed in a distance matrix, in which case
you don't even have to do that.

### Probability-based embedding

But you are probably carrying out a probability-based embedding. In this case
you have to generate the input probabilities from the input distances. Normally
the input kernel function is a gaussian, and there is a free parameter to tune
that control the spread of the function. Sometimes this parameter is defined
as the bandwidth or variance, and as it gets larger, the spread of the gaussian
increases. Other times you'll see the parameter referred to as the "precision", 
which is the inverse of the squared bandwidth, and hence the larger the 
precision is, the faster it decays. The `sneer` options refer to it 
as the precision, so I'll try not to refer to it in other ways, unless it makes 
the discussion clearer.

### Perplexity and precision

The precision of the kernel function is not set directly. It's done by 
specifying a desired value for the perplexity, which is usually described as 
being a continuous analog to the number of nearest neighbors of a point. The 
user sets the perplexity, and then the precision of the kernel is adjusted so 
that theprobability associated with each point generates the desired perplexity.

You may be wondering at this point if there's a free parameter associated with
the output kernel function. The answer is, for t-SNE, no. But for other
embedding methods there is. For example, the original SNE method also uses
a gaussian kernel for the output distances, but just sets all the bandwidths
to unity. For more exotic treatments, look for the discussion of the 
`prec_scale` parameter below.

### `perplexity`

Set it to the value you want for the perplexity:

```R
# set the perplexity to half the cluster size
s1k_tsne <- sneer(s1k, perplexity = 50)
```

The original t-SNE paper suggests a value of '5-50', and used a value of 40 for 
the data it presented results for, but you may want to experiment. The larger
the value of the perplexity, the flatter the probability distribution. Set it 
too high and you will end up with a homogeneous embedding which ignores local
structure. Set it too low and you'll get a series of overlapping, disconnected
regions.

### `perp_scale`

Apart from the annoyance of having to work out a good value of the perplexity,
some researchers have suggested that, especially when using a initialization
from a dense random distribution (see the 
[Output Initialization](output-initialization.html) page for more on that),
a larger perplexity would be useful initially to avoid falling into local
optima.

There have been a few suggestions made in the literature to ameliorate or
even remove these issues with the perplexity. The 
[NeRV](http://www.jmlr.org/papers/v11/venna10a.html) paper suggests starting 
the optimization with a fairly high perplexity value and then recalculating the
input probabilities at progressively lower perplexities as the optimization 
continues. Set the `perp_scale` parameter to `"step"` to try this. However,
you must now pass the `perplexity` parameter a vector containing the 
perplexities to use or rely on sneer choosing some reasonable values for you:

```R
# step perplexity from 150 to 50:
s1k_tsne <- sneer(s1k, perp_scale = "step", 
                  perplexity = c(150, 125, 100, 75, 50))
                  
# Uses a series of 5 reasonable perplexities based on data set size
s1k_tsne <- sneer(s1k, perp_scale = "step")
```

A related, but more complex approach is that described in the 
[multiscale SNE](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-64.pdf)
paper. Again, multiple perplexity values are tried, but in this case, the input
probabilities are averaged. Set `perp_scale` to `"multi"` to try this:

```R
s1k_tsne <- sneer(s1k, perp_scale = "multi")
```

Like the `"step"` scaling method, you can supply your own perplexities, but
the default values are reasonable: to attempt to cover multiple scales of the
data, perplexities in decreasing scales of two are tried.

Sneer is already not very speedy, be aware that using this perplexity 
averaging slows down the embedding further, because the output probabilities 
are also calculated multiple times and averaged. For the full effect, you 
should be using a method where the output kernel has a free parameter 
(see the `prec_scale` option below). If you are using t-SNE, using this method
causes the embedding to do a lot of un-necessary work: there is no free 
parameter in the output kernel, so identical probabilities are generated
multiple times and pointlessly averaged.

### `perp_scale_iter`

If you are using the stepped or multiscaled perplexities, the usual approach
is to slowly introduce the new perplexities over time, but not to take so long
that you don't have time to do any actual optimization. By default, the scaling
will take place over the first 20% of the optimization. You can change this
by setting the `perp_scale_iter` parameter to the number of iterations you
want the scaling to occur over:

```R
# make sure the perplexity scaling is finished by iteration 150
s1k_tsne <- sneer(s1k, perp_scale = "step", perp_scale_iter = 150)
```

The default number of total iterations is 1000, and is controlled by the
`max_iter` parameter. See the [Optimization](optimization.html) section if you 
want to know more.

### `prec_scale`

This option controls how the free parameter (if it exists) on the output kernel
is scaled, relative to the input kernel. The "prec" in the option name is short
for "precision", which as discussed above, is inversely related to the spread
of the kernel function.

SNE and its variants go to great pains to ensure that each point in the input
space has its own precision to match the density of points in its neighborhood.
But for the output space, the precisions are all set so the corresponding 
bandwidths are 1. And in the case of t-SNE, there isn't even a free parameter to 
adjust.

Some approaches take a different view. The 
[NeRV](http://www.jmlr.org/papers/v11/venna10a.html) paper suggests 
transferring the precisions from the input kernel to the output kernel.

This can be accomodated by using the `prec_scale` argument with the argument
`"t"` (for "transfer"):

```R
s1k_ssne <- sneer(s1k, prec_scale = "t", method = "ssne")
```

Note also that I've explicitly changed the embedding `method` from t-SNE to 
the related SSNE method, which has a free parameter on the output kernel to
make use of the precisions. For more on the `method` parameter, see the section
on [Embedding Methods](embedding-methods.html).

In conjunction with the multiscale perplexity scaling, a more complex approach 
to the output kernel scaling is advocated in the 
[multiscale JSE](http://dx.doi.org/10.1016/j.neucom.2014.12.095) paper (JSE
is an embedding method similar to t-SNE and SSNE, so can be applied other
methods).

This involves taking "intrinsic dimensionality" values (which you can see 
summaries of logged to the console during initialization) and using them
to scale the output kernel precision for each perplexity. See the "Console
Output" section below for a bit more explanation of intrinsic dimensionality,
but the full procedure is sufficiently complex that you should read the 
multscale JSE paper if you want the gory details. To just turn it on and
try it out, set the `prec_scale` parameter to `"s"` (for "scale").

For this setting to do anything, you must be using an embedding method which 
has a free parameter in the output kernel (i.e. not t-SNE). Additionally, you 
should be using multiple perplexities by setting the `perp_scale` parameter 
to `"step"` or  `"multi"`. Otherwise, although the output kernel precision is 
changed,  all kernels get the same value, and this will have no effect on the 
relative distances in the final configuration.

An example of a valid combination is:

```R
s1k_mssne <- sneer(s1k, perp_scale = "multi", prec_scale = "s", method = "ssne")
```

### `perp_kernel_fun`

Some methods directly or indirectly use a sparse representation of the input
probabilities, which substantially helps with memory and speed issues. See,
for instance, the [Spectral Directions](https://arxiv.org/abs/1206.4646) and
[ws-SNE (PDF)](http://jmlr.org/proceedings/papers/v32/yange14.pdf) papers.

Sneer doesn't support any sparse representations, but if you want to see what 
the effect of replacing the gaussian kernel with a step-function kernel 
(which effectively sets all the non-nearest neighbor probabilities to zero), 
you can do so:

```R
# Only the 50 nearest neighbors of a point will have a non-zero probability
s1k_tsne <- sneer(s1k, perplexity = 50, perp_scale_fun = "step")
```

But let me stress once again that you don't get any sparsity advantages from
doing this, so it's probably only of academic interest.

If you have an interest in the "intrinsic dimensionality" that is logged to
the console during input initialization (see the 'Console Output' section
below), don't use this kernel function, as the intrinsic dimensionality is only 
defined for gaussian functions.

### `exaggerate`

The early exaggeration "trick" is akin to perplexity scaling in the sense
that it causes the input probabilities to change. Set the `exaggerate` 
parameter to the value that the input probabilities should be multiplied by
(normally `4`). The scaling will be turned off at the iteration number given
by `exaggerate_off_iter` (normally `50` or `100`).

```R
s1k_tsne <- sneer(s1k, exaggerate = 4, exaggerate_off_iter = 100)
```

The effect of the exaggeration is to encourage tight, widely-spaced clusters,
so it will disrupt the initial configuration. This makes sense if you are 
initializing from random, but may make less sense for a PCA initialization
or initializing from a fixed set of coordinates. See the use of `init`
parameter in [Output Initialization](output-initialization.html) for more on
output initialization.

### Console Output

The results of the perplexity scaling will result in some summary statistic
being logged to the console. They'll look a bit like this:

```
Parameter search for perplexity = 32
prec: Min. : 1.493 |1st Qu. : 3.377 |Median : 4.077 |Mean : 4.149 |3rd Qu. : 4.753 |Max. : 10.12 |
P: Min. : 0 |1st Qu. : 2.233e-09 |Median : 2.355e-07 |Mean : 0.001 |3rd Qu. : 1.449e-05 |Max. : 0.3655 |
dims: Min. : 2.306 |1st Qu. : 4.133 |Median : 4.813 |Mean : 4.941 |3rd Qu. : 5.56 |Max. : 9.686 |
```

This gives the results for setting the perplexity to 32 for the `s1k` data set.
`prec` provide summaries of the precisions calculated for each input 
observation. Different implementations of t-SNE and related embedding methods
can define the parameter associated with the gaussian in different ways. 
In `sneer`, the gaussian is defined as:

$$e^{-\tau x^2 / 2} \equiv e^{-\beta x^2} \equiv e^{-x^2 / 2\sigma^2}$$

where $\tau$ is the precision and $\sigma$ is the bandwidth. The form with
$\beta$ in is the most convenient to use computationally and that's what you'll
see if you go looking in the source code for t-SNE implementations. 
Unfortunately there isn't a convenient name for $\beta$. Informally, you'll see 
it referred to as the precision too, but let's agree to call it the 
"half-precision" if we ever need to refer to it again where the distinction
is important. Oh, and sometimes $\sigma$ is reported with the factor of two 
folded into it. This can make comparing output between different programs
difficult!

`P` is a summary of the resulting probabilities. Mainly it can help you ensure
that the normalization procedure is doing what you think (there are two main
ways to normalize the probabilities, see the [Gradients](gradients.html) page
for more.

Finally, the `dims` line summarizes the "intrinsic dimensionality", which is
used in the multiscale JSE embedding method (see the section on the 
`prec_scale` for more on that). Briefly, the input probability and perplexities 
are used to estimate the dimensionality of the input space neighborhood around 
each point. For example, if the data is distributed in a 2D gaussian in a 10D 
space, you would expect the intrinsic dimensionality of each point to be closer 
to 2 than 10.

Edge effects, data set size, clustering and sparseness of the neighborhood 
can have an effect on the reported values, but there might be some value in
comparing the distribution between different data sets. It can also be
instructive to play around with the simulation data sets that are part of
the `snedata` package which have known topology and dimensionality and seeing
what the reported intrinsic dimensionality is. For more on the `snedata` package
see the [Data Sets](datasets.html) section. And for the definition and use
of intrinsic dimensionality, see the 
[multiscale JSE](http://dx.doi.org/10.1016/j.neucom.2014.12.095) paper.

Previous: [Preprocessing](preprocessing.html). Next: [Output Initialization](output-initialization.html). Up: [Index](index.html).
