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
# sneer function knows how to do lots of embedding
?sneer
```

### Examples
```R
# PCA on iris dataset and plot result using Species label name
res <- sneer(iris, indexes = 1:4, label_name = "Species", method = "pca")

# Same as above, but with sensible defaults (use all numeric columns, plot
# with first factor column found)
res <- sneer(iris, method = "pca")

# scale columns so each one has mean 0 and variance 1
res <- sneer(iris, method = "pca", scale_type = "a")

# full species name on plot is cluttered, so just use the first two
# letters and half size
res <- sneer(iris, method = "pca", scale_type = "a", label_chars = 2,
             label_size = 0.5)

# You need to install and load ggplot2 and RColorBrewer yourself
library(ggplot2)
library(RColorBrewer)
# Optionally use ggplot2 and RColorBrewer palettes for the plot
res <- sneer(iris, method = "pca", scale_type = "a", plot_type = "g")

# Use a different ColorBrewer palette, bigger points, and range scale each
# column
res <- sneer(iris, method = "pca", scale_type = "r", plot_type = "g",
             palette = "Dark2", label_size = 2)

# metric MDS starting from the PCA
res <- sneer(iris, method = "mmds", scale_type = "a", init = "p")

# Sammon map starting from random distribution
res <- sneer(iris, method = "sammon", scale_type = "a", init = "r")

# TSNE with a perplexity of 32, initialize from PCA
res <- sneer(iris, method = "tsne", scale_type = "a", init = "p",
             perplexity = 32)
# default settings are to use TSNE with perplexity 32 and initialization
# from PCA so the following is the equivalent of the above
res <- sneer(iris, scale_type = "a")

# NeRV method, starting at a more global perplexity and slowly stepping
# towards a value of 32 (might help avoid local optima)
res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step")

# NeRV method has a lambda parameter - closer to 1 it gets, the more it
# tries to avoid false positives (close points in the map that aren't close
# in the input space):
res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step",
             lambda = 1)

# Original NeRV paper transferred input exponential similarity kernel
# precisions to the output kernel, and initialized from a uniform random
# distribution
res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step",
             lambda = 1, prec_scale = "t", init = "u")

# Like NeRV, the JSE method also has a controllable parameter that goes
# between 0 and 1, called kappa. It gives similar results to NeRV at 0 and
# 1 but unfortunately the opposite way round! The following gives similar
# results to the NeRV embedding above:
res <- sneer(iris, scale_type = "a", method = "jse", perp_scale = "step",
             kappa = 0)

# Rather than step perplexities, use multiscaling to combine and average
# probabilities across multiple perplexities. Output kernel precisions
# can be scaled based on the perplexity value (compare to NeRV example
# which transferred the precision directly from the input kernel)
res <- sneer(iris, scale_type = "a", method = "jse", perp_scale = "multi",
             prec_scale = "s")

# HSSNE has a controllable parameter, alpha, that lets you control how
# much extra space to give points compared to the input distances.
# Setting it to 1 is equivalent to TSNE, so 1.1 is a bit of an extra push:
res <- sneer(iris, scale_type = "a", method = "hssne", alpha = 1.1)

# wTSNE treats the input probability like a graph where the probabilities
# are weighted edges and adds extra repulsion to nodes with higher degrees
res <- sneer(iris, scale_type = "a", method = "wtsne")

# can use a step-function input kernel to make input probability more like
# a k-nearest neighbor graph (but note that we don't take advantage of the
# sparsity for performance purposes, sadly)
res <- sneer(iris, scale_type = "a", method = "wtsne", perp_kernel_fun = "step")

# Some quality measures are available to quantify embeddings
# The area under the RNX curve measures whether neighbors in the input
# are still neighors in the output space
res <- sneer(iris, scale_type = "a", method = "wtsne",
             quality_measures =  c("n"))

# If your dataset labels divide the data into natural classes, can calculate 
# average area under the ROC and/or precision-recall curve too, but you need to 
# have installed the PRROC package. And all these techniques can be slow (scale 
# with the square of the number of observations).
library(PRROC)
res <- sneer(iris, scale_type = "a", method = "wtsne", 
             quality_measures =  c("n", "r", "p"))
             
# export the distance matrices and do whatever quality measures we want at our 
# leisure
res <- sneer(iris, scale_type = "a", method = "wtsne", ret = c("dx", "dy"))

# export degree centrality, input weight function precision parameters,
# and intrinsic dimensionality
res <- sneer(iris, scale_type = "a", method = "wtsne", 
  ret = c("deg", "prec", "dim"))

# Plot the embedding as points colored by category, using the rainbow
# palette:
embed_plot(res$coords, iris$Species, palette = "rainbow")

# Load the RColorBrewer Library
library(RColorBrewer)

# Use a Color Brewer Qualitative palette
embed_plot(res$coords, iris$Species, palette = "Dark2")

# Visualize embedding colored by various values:
# Degree centrality
embed_plot(res$coords, x = res$deg)
# Intrinsic Dimensionality using the PRGn palette 
# (requires RColorBrewer package to be installed)
embed_plot(res$coords, x = res$dim, palette = "PRGn")
# Input weight function precision parameter with the Spectral palette
embed_plot(res$coords, x = res$prec, palette = "Spectral")
# Calculate the 32-nearest neighbor preservation for each observation
# 0 means no neighbors preserved, 1 means all of them
pres32 <- nbr_pres(res$dx, res$dy, 32)
embed_plot(res$coords, x = pres32, cex = 1.5)

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

* Embedding with [t-SNE](http://jmlr.org/papers/v9/vandermaaten08a.html) 
and its variants ASNE and SSNE.
* Sammon mapping and metric Multidimensional Scaling.
* [Heavy-Tailed Symmetric SNE](http://papers.nips.cc/paper/3770-heavy-tailed-symmetric-stochastic-neighbor-embedding) (HSSNE).
* [Neighbor Retrieval Visualizer](http://www.jmlr.org/papers/v11/venna10a.html) (NeRV).
* [Jensen-Shannon Embedding](http://www.sciencedirect.com/science/article/pii/S0925231213001471) (JSE).
* [Multiscale SNE](http://www.sciencedirect.com/science/article/pii/S0925231215003641) (msSNE).
* [Weighted SNE using degree centrality (PDF)](http://www.jmlr.org/proceedings/papers/v32/yange14.pdf) (wSSNE).
* Nesterov Accelerated Gradient method for optimization.
* The usual t-SNE Steepest descent with momentum and Jacobs adaptive step size
if NAG is too racy for you.
* The bold driver adaptive step algorithm if you want to mix it up a little.
* The [Spectral Directions (PDF)](http://faculty.ucmerced.edu/mcarreira-perpinan/papers/icml12.pdf)
optimization method of Vladymyrov and Carreira-Perpiñán, although in a 
non-sparse form.
* The L-BFGS method from the R `optim` function.
* Output initialization options include using PCA scores matrix for easier
reproducibility.
* Various simple preprocessing options.
* Numerical scores for qualitatively evaluating the embedding.
* s1k, a small (1000 points) 9-dimensional synthetic dataset that exemplifies
the "crowding problem".

If you install and load the [rcgmin](https://github.com/jlmelville/rcgmin) 
package:
```R
devtools::install_github("jlmelville/rcgmin")
library("rcgmin")
```

You can also access some extra optimization options:

* A Polak-Ribiere-style conjugate gradient optimizer.
* A choice of the More-Thuente or Rasmussen line search algorithm when using
CG, spectral direction or NAG.

Again, see the documentation and examples for the `embed` function for more 
details.

### Limitations and Issues
* It's in pure R, so it's slow. 
* It doesn't implement any of the Barnes-Hut or multipole or related approaches
to speed up the distance calculations from O(N^2), so it's slow.
* It doesn't work with sparse matrices... so it's slow and it can't work with
large datasets.

Consider this package designed for experimenting on smaller datasets, not 
production-readiness.

Also, fitting everything I wanted to do into one package has involved 
splitting everything up into lots of little functions, so good luck finding 
where anything actually gets done. Thus, its pedagogical value is negligible, 
unless you were looking for an insight into my questionable design, naming and 
decision making skills. But this is a hobby project, so I get to make it as 
over-engineered as I want.

### See also
I have some other packages that create or download datasets often used in 
SNE-related research: 
[Simulation, Olivetti and Frey Faces](https://github.com/jlmelville/snedata), 
[COIL-20](https://github.com/jlmelville/coil20), and 
[MNIST Digit](https://github.com/jlmelville/mnist).

### Acknowledgements
I reverse engineered some specifics of the Spectral Directions gradient by 
translating the relevant part of the Matlab implementation provided on the 
Carreira-Perpiñán group's 
[software page](http://faculty.ucmerced.edu/mcarreira-perpinan/software.html).
Professor Carreira-Perpiñán kindly agreed to allow the resulting R code to
be under the GPL license of this package. Obviously, assume any mistakes, errors
or resulting destruction of your computer is a bug in sneer.

### License
[GPLv2 or later](https://www.gnu.org/licenses/gpl-2.0.txt).
