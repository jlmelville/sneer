# sneer

Stochastic Neighbor Embedding Experiments in R

An R package for experimenting with dimensionality reduction techniques, 
including the popular 
[t-Distributed Stochastic Neighbor Embedding](https://lvdmaaten.github.io/tsne/)
(t-SNE).

## Installing

```R
# install.packages("devtools")
devtools::install_github("jlmelville/sneer")
```

## Documentation

```R
package?sneer
# sneer function knows how to do lots of embedding
?sneer
```

Also see the (currently under-construction) 
[documentation web pages](http://jlmelville.github.io/sneer/) for
a more detailed explanation.

## Examples

```R
# t-SNE on the iris dataset:
res <- sneer(iris)
# then do what you want with the embedded coordinates in res$coords

# sneer does t-SNE, looks for numeric columns and a factor column to color 
# points with automatically, and does tSNE by default, but you can get specific:
res <- sneer(iris[, 1:4], label = iris$Species, method = "tsne", 
             perplexity = 25)
```

There is a section of the documentation that has (many) more 
[examples](http://jlmelville.github.io/sneer/examples.html).

## Motivation

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

## Features

Currently sneer offers:

* Embedding with [t-SNE](http://jmlr.org/papers/v9/vandermaaten08a.html) 
and its variants ASNE and SSNE.
* Sammon mapping and metric Multidimensional Scaling.
* [Heavy-Tailed Symmetric SNE](http://papers.nips.cc/paper/3770-heavy-tailed-symmetric-stochastic-neighbor-embedding) (HSSNE).
* [Neighbor Retrieval Visualizer](http://www.jmlr.org/papers/v11/venna10a.html) (NeRV).
* [Jensen-Shannon Embedding](http://www.sciencedirect.com/science/article/pii/S0925231213001471) (JSE).
* [Multiscale SNE](http://www.sciencedirect.com/science/article/pii/S0925231215003641) (msSNE).
* [Weighted SNE using degree centrality](http://www.jmlr.org/proceedings/papers/v32/yange14.html) (ws-SSNE).
* A variety of optimizations using the [mize](https://github.com/jlmelville/mize) package.
* The [Spectral Directions](https://arxiv.org/abs/1206.4646)
optimization method of Vladymyrov and Carreira-Perpiñán, although in a 
non-sparse form.
* Output initialization options include using PCA scores matrix for easier
reproducibility.
* Various simple preprocessing options.
* Numerical scores for qualitatively evaluating the embedding.
* s1k, a small (1000 points) 9-dimensional synthetic dataset that exemplifies
the "crowding problem".

## Limitations and Issues

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

## See also

I have some other packages that create or download datasets often used in 
SNE-related research: 
* [Simulation, Olivetti and Frey Faces](https://github.com/jlmelville/snedata), 
* [COIL-20](https://github.com/jlmelville/coil20) 
* [MNIST Digit](https://github.com/jlmelville/mnist)
* [mize](https://github.com/jlmelville/mize), the optimization package.

## Acknowledgements

I reverse engineered some specifics of the Spectral Directions gradient by 
translating the relevant part of the Matlab implementation provided on the 
Carreira-Perpiñán group's 
[software page](http://faculty.ucmerced.edu/mcarreira-perpinan/software.html).
Professor Carreira-Perpiñán kindly agreed to allow the resulting R code to
be under the GPL license of this package. Obviously, assume any mistakes, errors
or resulting destruction of your computer is a bug in sneer.

## License

[GPLv2 or later](https://www.gnu.org/licenses/gpl-2.0.txt). The optimization
part of sneer is provided by the [mize](https://github.com/jlmelville/mize)
package, which is available under the 
[BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) license.

