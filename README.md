# sneer
Stochastic Neighbor Embedding Experiments in R

An R package for experimenting with dimensionality reduction techniques, 
including the popular t-Distributed Stochastic Neighbor Embedding (t-SNE).

### Installing:
```R
install.packages("devtools")
devtools::install_github("jlmelville/sneer")
```

### Documentation:
```R
# Package documentation
package?sneer

# Embedding function documentation
?embed_sim
```

### Example
```
# t-SNE on the iris using parameters from the t-SNE paper
tsne_iris <- embed_sim(iris[, 1:4],
               method = tsne(),
               opt = tsne_opt(),
               init_inp = make_init_inp(perplexity = 25),
               init_out = make_init_out(stdev = 1e-4),
               tricks = tsne_tricks(),
               reporter = make_reporter(
                 plot_fn = make_plot(iris, "Species", make_label(2))))
```
There are more examples in the documentation.

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

Sneer is very new so there's not a lot here that's isn't available in other
packages right now. But currently it offers:

* Embedding with t-SNE and its variants ASNE and SSNE.
* Nesterov Accelerated Gradient method for optimization.
* The usual t-SNE Steepest descent with momentum and Jacobs adaptive step size
if NAG is too racy for you.
* The bold driver adaptive step algorithm if you want to mix it up a little.
* Output initialization options include using PCA scores matrix for easier
reproducibility.
* Various simple preprocessing options.
* s1k, a small (1000 points) 9-dimensional synthetic dataset that exemplifies
the "crowding problem".
* Some tests! (Not many, though)
* A bit of documentation!

### Roadmap
Some new algorithms coming soon:

* Heavy-Tailed Symmetric SNE (HSSNE).
* Neighbor Retrieval Visualizer (NeRV).
* Jensen-Shannon Embedding (JSE).
* MDS and Sammon Mapping for completeness' sake.
* Numerical scores for qualitatively evaluating the embedding.

I've implemented the above already, I'm just too ashamed of the terrible code it
involves to air it in public. But unless I get hit by a bus, this will actually
happen. Other things I'd like to do:

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
