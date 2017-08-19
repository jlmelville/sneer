---
title: "Output Initialization"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---
Previous: [Input Initialization](input-initialization.html). Next: [Optimization](optimization.html). Up: [Index](index.html).

In contrast to input initialization, output initialization is pretty 
straightforward. You just need a source of initial coordinates, which you
can specify with the `init` parameter.

## `init`

A common approach to initialization is to start with a random configuration.
The most important thing to do is to make sure the points aren't too far away
initially, or otherwise the gradients are very small and no optimization occurs.

### Gaussian Random

The [t-SNE paper](http://jmlr.org/papers/v9/vandermaaten08a.html) suggests
initialization from a small (standard deviation of `1e-4`) gaussian 
distribution. Set `init` to `"random"` for that:

```R
set.seed(1337) # if you want to be reproducible
s1k_tsne <- sneer(s1k, init = "random")
```

### Uniform Random

The [NeRV paper](http://www.jmlr.org/papers/v11/venna10a.html) uses a uniform
distribution instead. I can't imagine it makes much difference, but that's what
sneer is for:

```R
s1k_tsne <- sneer(s1k, init = "uniform")
```

These small random distributions do well with probability-based embeddings, but 
they produce pretty horrible results for Sammon Mapping.

### PCA

For Sammon Mapping, and in my opinion most embedding methods, a better choice 
is the default initialization method, which uses PCA. It's the default because 
it's my personal preference, but it's also used for initialization in the 
[JSE paper](http://dx.doi.org/10.1016/j.neucom.2012.12.036)). The first two 
score vectors (principal components) from a PCA of the input data is used. No
scaling is done to the data, although it is centered.

```R
s1k_tsne <- sneer(s1k, init = "pca")
s1k_tsne <- sneer(s1k)  # same thing as the above
```

If you provide a distance matrix instead of a data frame, classical MDS is 
carried out instead of PCA, which is effectively the same thing as PCA if
the input distances are Euclidean.

This removes any niggling issues of reproducibility and in my experience gives 
good results for most data sets without the hassle of having to repeat the 
embedding multiple times.

PCA is not always better than a random initialization and if the scores plot
results in some distant points, these will have a hard time having any 
meaningful gradient associated with them, at least with methods like t-SNE. But
it's always worth trying first.

### Scaled PCA

Large distances are bad news for SNE-family gradients: squared distances 
combined with an exponential kernel in particular can easily lead to numeric
underflow and a weight matrix made up of all zeros. And there's no reason that 
PCA on unscaled input data can't result in large distances. I have experienced
this problem first hand with multiple datasets initialized with PCA.

The scaled PCA initialization attempts to solve this problem, by rescaling the 
PCA output so that the standard deviations of the scores vectors are suitably 
small. `sneer` uses a value of `1e-4`, just like the Gaussian random 
initialization, but with the added bonus of being deterministic without relying 
on setting a specific random seed (which doesn't work across different
architectures anyway).

Best of both worlds? You be the judge. This is not the default initialization,
because I have not seen this form of initialization anywhere in the literature.
But I recommend it highly if you won't be scaling or otherwise pre-processing
your input data.

```R
s1k_tsne <- sneer(s1k, init = "spca")
```

### Existing Coordinates

Finally, if you already have some coordinates to hand, you can provide them 
directly:

```R
input_coords <- some_other_initialization_method(s1k)
s1k_tsne <- sneer(s1k, init = input_coords)
```

In the example above, `input_coords` should be a matrix with dimensions n x 2,
where n is the number of rows in `s1k`, which happens to be 1000.

Previous: [Input Initialization](input-initialization.html). Next: [Optimization](optimization.html). Up: [Index](index.html).

