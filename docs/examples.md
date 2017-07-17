---
title: "Examples"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---

Provides a sampling of what's available through `sneer`. But see
the full [Documentation](index.html) for the available options.

## Run t-SNE By Default
```R
res <- sneer(iris)
```

## PCA on iris dataset and plot result using Species label name

```R
res <- sneer(iris, indexes = 1:4, label_name = "Species", method = "pca")

# Same as above, but with sensible defaults (use all numeric columns, plot
# with first factor column found)
res <- sneer(iris, method = "pca")
```

## Scaling

### Scale Columns to Mean 0 and Unit Variance

```R
res <- sneer(iris, method = "pca", scale_type = "a")
```

### Range Scale Each Column
```R
res <- sneer(iris, method = "pca", scale_type = "r")
```

### Range Scale Entire Matrix
```R
res <- sneer(iris, method = "pca", scale_type = "m")
```

### Only Run Plot for 200 Iterations
```R
res <- sneer(iris, max_iter = 200)
```

## Visualization During the Embedding

### Plotting Category Names
```R
# full species name on plot is cluttered, so just use the first two
# letters and half size
res <- sneer(iris, scale_type = "a", label_chars = 2, 
                   point_size = 0.5, plot_labels = TRUE)
```

### Use Pre-Chosen Colors in the Embedding Plot
```R
# Create a 5D gaussian with its own column specifying colors to use
# for each point (in this case, random)
g5d <- data.frame(matrix(rnorm(100 * 5), ncol = 5),
                  color = rgb(runif(100), runif(100), runif(100)),
                  stringsAsFactors = FALSE)
# Specify the name of the color column and the plot will use it rather than
# trying to map factor levels to colors
res <- sneer(g5d, method = "pca", color_name = "color")
```

### Use `ggplot2` for the Embedding Plot
```R
# You need to install and load ggplot2 and RColorBrewer yourself
library("ggplot2")
res <- sneer(iris, method = "pca", scale_type = "a", plot_type = "g")
```

### Use ColorBrewer color schemes for the embedding plot
```R
# Use a different ColorBrewer palette, bigger points, and range scale each
# column
res <- sneer(iris, method = "pca", scale_type = "r", plot_type = "g",
             color_scheme = "Dark2", label_size = 2)
```

### No plot at all
```R
res <- sneer(iris, plot_type = "n")
```

## Embedding Methods

### Metric MDS
```R
res <- sneer(iris, method = "mmds")
```

### Sammon Map
```R
# Sammon map starting from random distribution
res <- sneer(iris, method = "sammon", scale_type = "a", init = "r")
```

### t-SNE
```R
# TSNE with a perplexity of 32, initialize from PCA
res <- sneer(iris, method = "tsne", scale_type = "a", init = "p",
             perplexity = 32)
             
# default settings are to use TSNE with perplexity 32 and initialization
# from PCA so the following is the equivalent of the above
res <- sneer(iris, scale_type = "a")
```

### t-SNE with Jacobs Step Size method
```R
# Use the standard tSNE optimization method (Jacobs step size method) with
# step momentum. Range scale the matrix and use an aggressive learning
# rate (eta).
res <- sneer(iris, scale_type = "m", perplexity = 25, opt = "tsne",
             eta = 500)
```

### t-SNE with Jacobs Step Size and Early Exaggeration
```R
res <- sneer(iris, scale_type = "m", perplexity = 25, opt = "tsne",
             eta = 500, exaggerate = 4, exaggerate_off_iter = 100)
```

### t-SNE with Jacobs Step Size, Early Exaggeration and Random Initialization

```R
# By default we initialize from a PCA scores plot, but we can initialize
# from random too
res <- sneer(iris, scale_type = "m", perplexity = 25, opt = "tsne",
             eta = 500, exaggerate = 4, exaggerate_off_iter = 100,
             init = "r")
```

### NeRV with perplexity stepping

```R
# Will step from a global-ish perplexity value towards a perplexity of 32
res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step")
```

### NeRV with adjusted lambda parameter
```R
# NeRV method has a lambda parameter - closer to 1 it gets, the more it
# tries to avoid false positives (close points in the map that aren't close
# in the input space):
res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step",
             lambda = 1)
```

### NeRV with transferred input kernel precisions
```R
# Original NeRV paper transferred input exponential similarity kernel
# precisions to the output kernel, and initialized from a uniform random
# distribution
res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step",
             lambda = 1, prec_scale = "t", init = "u")
```

### JSE 
```R
# Like NeRV, the JSE method also has a controllable parameter that goes
# between 0 and 1, called kappa. It gives similar results to NeRV at 0 and
# 1 but unfortunately the opposite way round! The following gives similar
# results to the NeRV embedding above:
res <- sneer(iris, scale_type = "a", method = "jse", perp_scale = "step",
             kappa = 0)
```

### Multiscale JSE
```R
# Rather than step perplexities, use multiscaling to combine and average
# probabilities across multiple perplexities. Output kernel precisions
# can be scaled based on the perplexity value (compare to NeRV example
# which transferred the precision directly from the input kernel)
res <- sneer(iris, scale_type = "a", method = "jse", perp_scale = "multi",
             prec_scale = "s")
```

### Heavy-tailed SNE
```R
# HSSNE has a controllable parameter, alpha, that lets you control how
# much extra space to give points compared to the input distances.
# Setting it to 1 is equivalent to TSNE, so 1.1 is a bit of an extra push:
res <- sneer(iris, scale_type = "a", method = "hssne", alpha = 1.1)
```

### Dynamic HSSNE
```R
# Similar to HSSNE, but directly optimizes alpha along with the coordinates.
# Parameters associated with kernel optimization are passed via a "dyn" list.
# We wait 25 iterations here before starting to modify alpha from its initial 
# value of 0, to allow the configuration to settle a little bit:
res <- sneer(iris, method = "dhssne", dyn = list(kernel_opt_iter = 25), alpha = 0)
```

### Inhomogenenous t-SNE
```R
# Similar to DHSSNE, but directly optimizes a per-point degree of freedom
# Also recommended to wait a few iterations before starting optimization of
# each dof value from initial value of 10:
res <- sneer(iris, method = "itsne", dyn = list(kernel_opt_iter = 25), dof = 10)
```

### Importance-weighted SNE
```R
# ws-SNE treats the input probability like a graph where the probabilities
# are weighted edges and adds extra repulsion to nodes with higher degrees
res <- sneer(iris, scale_type = "a", method = "wssne")
```

### Use step function for input probabilities
```R
# can use a step-function input kernel to make input probability more like
# a k-nearest neighbor graph (but note that we don't take advantage of the
# sparsity for performance purposes, sadly)
res <- sneer(iris, scale_type = "a", method = "wtsne", perp_kernel_fun = "step")
```

## Export Data For Later Analysis

```R
# export the distance matrices and do whatever quality measures we want at our
# leisure
res <- sneer(iris, scale_type = "a", method = "wtsne", ret = c("dx", "dy"))

# export degree centrality, input weight function precision parameters,
# and intrinsic dimensionality
res <- sneer(iris, scale_type = "a", method = "wtsne",
  ret = c("deg", "prec", "dim"))
```

## Quality Measures

If your dataset labels divide the data into natural classes, can calculate
average area under the ROC and/or precision-recall curve too, but you need to
have installed the PRROC package. 

You can also calculate the Area Under the RNX Curve, which is a measure of
neighborhood preservation between the upper and lower dimensions, which doesn't
need any categorization.

All these techniques can be slow (scale with the square of the number of 
observations).

```R
# "n" - RNX AUC
res <- sneer(iris, scale_type = "a", method = "wtsne",
             quality_measures =  c("n"))

# Install and load the PRROC package
library(PRROC)

# "r" - ROC AUC
# "p" - Precision Recall AUC
res <- sneer(iris, quality_measures =  c("n", "r", "p"))
```

## Quality Measures as Separate Functions
```R
# export input (dx) and output (dy) distance matrices
res <- sneer(iris, scale_type = "a", method = "wtsne",
             ret - c("dx", "dy"))
             
rnx_auc <- rnx_auc_embed(res$dx, res$dy)

pr_auc <- pr_auc_embed(res$dy, iris$Species)

roc_auc <- roc_auc_embed(res$dy, iris$Species)
```

## Visualizing the Results After Embedding

```R
res <- sneer(iris)

# Plot the embedding as points colored by category, using the rainbow
# color ramp function:
embed_plot(res$coords, iris$Species, color_scheme = rainbow)

# You can pass in the entire data frame and sneer will try and find a suitable
# column
embed_plot(res$coords, iris, color_scheme = rainbow)
```

### Using RColorBrewer Color Scheme Names
```R
res <- sneer(iris, scale_type = "a", method = "wtsne",
  ret = c("deg", "prec", "dim"))

# Load the RColorBrewer Library
library(RColorBrewer)

# Use a ColorBrewer Qualitative palette using the name
# NB: string not a function!
embed_plot(res$coords, iris$Species, color_scheme = "Dark2")
```

### Visualize Plot With Projected Quality Results

```R
res <- sneer(iris, scale_type = "a", method = "wtsne",
  ret = c("deg", "prec", "dim"))

# Load the RColorBrewer Library
library(RColorBrewer)

# Degree centrality
embed_plot(res$coords, res$deg)

# Intrinsic Dimensionality using the PRGn palette
# (requires RColorBrewer package to be installed)
embed_plot(res$coords, res$dim, color_scheme = "PRGn")

# Input weight function precision parameter with the Spectral palette
embed_plot(res$coords, res$prec, color_scheme = "Spectral")

# Calculate the 32-nearest neighbor preservation for each observation
# 0 means no neighbors preserved, 1 means all of them
pres32 <- nbr_pres(res$dx, res$dy, 32)
embed_plot(res$coords, pres32, cex = 1.5)
```

An inhomogeneous t-SNE example:

```R
# dof = 50, initial degrees of freedom for each point more Gaussian-like than
# t-distributed
# kernel_opt_iter = 0, don't wait any interations to start optimizing the d.o.f.
# ret = c("dyn"), export the optimized d.o.f.
iris_itsne <- sneer(iris, method = "itsne",  kernel_opt_iter = 0, dof = 50, 
                    ret = c("dyn"))
library(RColorBrewer)
# View the d.o.f on the output coordinates
# Dark blue means that point didn't need to stretch its output distances much
# compared to the input distances. Lighter colors indicate those that used
# a t-SNE-like stretching
embed_plot(iris_itsne$coords, iris_itsne$dyn$dof, color_scheme = "Blues")
```

### View Embeddings with Plotly

Using the `embed_plotly` function will open a browser window (unless you're
using RStudio, in which case it will appear in the Plots tab as usual).

```R
# You need to install and load plotly yourself
library("plotly")

res <- sneer(iris, ret = c("deg", "prec", "dim"))
  
# Get a nice legend if you use categorical values to map colors
embed_plotly(res$coords, iris)

# Get a color scale if you use a numeric vector
embed_plotly(res$coords, res$dim)

# Can use different color schemes
embed_plotly(res$coords, iris, color_scheme = topo.colors)

# Can Use ColorBrewer names too
embed_plotly(res$coords, res$dim, color_scheme = "Blues")
```
