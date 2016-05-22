# sneer
Stochastic Neighbor Embedding Experiments in R

An R package for experimenting with dimensionality reduction techniques, 
including the popular t-Distributed Stochastic Neighbor Embedding (t-SNE).

### Installing:
```R
# install.packages("devtools")
devtools::install_github("jlmelville/sneer")
```

### Documentation:
```R
package?sneer
?embed
```

### Examples
```R
# PCA on iris dataset and plot result using Species label name
res <- embed(iris, indexes = 1:4, label_name = "Species", method = "pca")

# Same as above, but with sensible defaults (use all numeric columns, plot
# with first factor column found)
res <- embed(iris, method = "pca")

# scale columns so each one has mean 0 and variance 1
res <- embed(iris, method = "pca", scale_type = "a")

# full species name on plot is cluttered, so just use the first two
# letters and half size
res <- embed(iris, method = "pca", scale_type = "a", label_chars = 2,
             label_size = 0.5)

# You need to install and load ggplot2 and RColorBrewer yourself
library(ggplot2)
library(RColorBrewer)
# Optionally use ggplot2 and RColorBrewer palettes for the plot
res <- embed(iris, method = "pca", scale_type = "a", plot_type = "g")

# Use a different ColorBrewer palette, bigger points, and range scale each
# column
res <- embed(iris, method = "pca", scale_type = "r", plot_type = "g",
             palette = "Dark2", label_size = 2)

# metric MDS starting from the PCA
res <- embed(iris, method = "mmds", scale_type = "a", init = "p")

# Sammon map starting from random distribution
res <- embed(iris, method = "sammon", scale_type = "a", init = "r")

# TSNE with a perplexity of 32, initialize from PCA
res <- embed(iris, method = "tsne", scale_type = "a", init = "p",
             perplexity = 32)
# default settings are to use TSNE with perplexity 32 and initialization
# from PCA so the following is the equivalent of the above
res <- embed(iris, scale_type = "a")

# NeRV method, starting at a more global perplexity and slowly stepping
# towards a value of 32 (might help avoid local optima)
res <- embed(iris, scale_type = "a", method = "nerv", perp_scale = "step")

# NeRV method has a lambda parameter - closer to 1 it gets, the more it
# tries to avoid false positives (close points in the map that aren't close
# in the input space):
res <- embed(iris, scale_type = "a", method = "nerv", perp_scale = "step",
             lambda = 1)

# Original NeRV paper transferred input exponential similarity kernel
# precisions to the output kernel, and initialized from a uniform random
# distribution
res <- embed(iris, scale_type = "a", method = "nerv", perp_scale = "step",
             lambda = 1, prec_scale = "t", init = "u")

# Like NeRV, the JSE method also has a controllable parameter that goes
# between 0 and 1, called kappa. It gives similar results to NeRV at 0 and
# 1 but unfortunately the opposite way round! The following gives similar
# results to the NeRV embedding above:
res <- embed(iris, scale_type = "a", method = "jse", perp_scale = "step",
             kappa = 0)

# Rather than step perplexities, use multiscaling to combine and average
# probabilities across multiple perplexities. Output kernel precisions
# can be scaled based on the perplexity value (compare to NeRV example
# which transferred the precision directly from the input kernel)
res <- embed(iris, scale_type = "a", method = "jse", perp_scale = "multi",
             prec_scale = "s")

# HSSNE has a controllable parameter, alpha, that lets you control how
# much extra space to give points compared to the input distances.
# Setting it to 1 is equivalent to TSNE, so 1.1 is a bit of an extra push:
res <- embed(iris, scale_type = "a", method = "hssne", alpha = 1.1)

# wTSNE treats the input probability like a graph where the probabilities
# are weighted edges and adds extra repulsion to nodes with higher degrees
res <- embed(iris, scale_type = "a", method = "wtsne")

# can use a step-function input kernel to make input probability more like
# a k-nearest neighbor graph (but note that we don't take advantage of the
# sparsity for performance purposes, sadly)
res <- embed(iris, scale_type = "a", method = "wtsne", perp_kernel_fun = "step")

# Some quality measures are available to quantify embeddings
# The area under the RNX curve measures whether neighbors in the input
# are still neighors in the output space
res <- embed(iris, scale_type = "a", method = "wtsne",
             quality_measures =  c("n"))

# If your dataset labels divide the data into natural classes, can calculate 
# average area under the ROC and/or precision-recall curve too, but you need to 
# have installed the PRROC package. And all these techniques can be slow (scale 
# with the square of the number of observations).
library(PRROC)
res <- embed(iris, scale_type = "a", method = "wtsne", 
             quality_measures =  c("n", "r", "p"))
             
# export the distance matrices and do whatever quality measures we want at our 
# leisure
res <- embed(iris, scale_type = "a", method = "wtsne", ret = c("dx", "dy"))

# calculate the 32-nearest neighbor preservation for each observation
# 0 means no neighbors preserved, 1 means all of them
pres32 <- nbr_pres(res$dx, res$dy, 32)

# use map2color helper function with diverging or sequential color palettes
# to map values onto the embedded points
plot(res$coords, col = map2color(pres32), pch = 20, cex = 1.5)

# export degree centrality, input weight function precision parameters,
# and intrinsic dimensionality
res <- embed(iris, scale_type = "a", method = "wtsne", 
  ret = c("deg", "prec", "dim"))

# Visualize embedding colored by various values (function requires RColorBrewer
# package to be installed):
# Degree centrality
embed_quant_plot(res$coords, res$deg)
# Intrinsic Dimensionality using the PRGn palette
embed_quant_plot(res$coords, res$dim, name = "PRGn")
# Input weight function precision parameter with the Spectral palette
embed_quant_plot(res$coords, res$prec, name = "Spectral")
```

### Motivation

There are a lot of dimensionality reduction techniques out there, and many that 
take inspiration from t-SNE, but understanding what makes them work (or not) is 
complicated by the differences in dataset preparation, preprocessing, output 
initialization, optimization, and other heuristics. 

Sneer is my attempt to write a package that not only provides a way to run 
multiple embedding algorithms with complete control over all the various 
twiddly bits, but also exposed lots of twiddly bits to twiddle on if that was 
what you wanted to do (and I do).

Its basic code was based heavily on Justin Donaldson's 
[tsne R package](https://github.com/cran/tsne), but is now mangled so far 
beyond its original form that I've made it a separate project rather than a 
fork. It does, however, inherit its license (GPL-2 or later).

### Features

Currently sneer offers:

* Embedding with t-SNE and its variants ASNE and SSNE.
* Sammon mapping and metric Multidimensional Scaling.
* Heavy-Tailed Symmetric SNE (HSSNE).
* Neighbor Retrieval Visualizer (NeRV).
* Jensen-Shannon Embedding (JSE).
* Multiscale SNE (msSNE).
* Weighted SNE using degree centrality (wSSNE).
* Nesterov Accelerated Gradient method for optimization.
* The usual t-SNE Steepest descent with momentum and Jacobs adaptive step size
if NAG is too racy for you.
* The bold driver adaptive step algorithm if you want to mix it up a little.
* Output initialization options include using PCA scores matrix for easier
reproducibility.
* Various simple preprocessing options.
* Numerical scores for qualitatively evaluating the embedding.
* s1k, a small (1000 points) 9-dimensional synthetic dataset that exemplifies
the "crowding problem".

### Roadmap
* Better documentation of internals so a hypothetical person who isn't me
could implement an embedding algorithm.
* Some vignettes exploring aspects of embedding.

### Limitations and Issues
It's in pure R, so it's slow. It's definitely designed for experimenting on 
smaller datasets, not production-readiness.

Also, fitting everything I wanted to do into one package has involved 
splitting everything up into lots of little functions, so good luck finding 
where anything actually gets done. Thus, its pedagogical value is negligible, 
unless you were looking for an insight into my questionable design, naming and 
decision making skills. But this is a hobby project, so I get to make it as 
over-engineered as I want.

### License
[GPLv2 or later](https://www.gnu.org/licenses/gpl-2.0.txt).
