---
title: "Preprocessing"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---

Previous: [Data Sets](datasets.html). Next: [Input Initialization](input-initialization.html). Up: [Index](index.html).

Sneer doesn't provide that much in the way of options for preprocessing, but
it's useful to have some of this exposed because it can affect the final
look of the embedding and you may need to scale data in a certain way if you're
comparing with other embedding methods.

The following options mainly apply to input supplied as a data frame. If you
supply a distance matrix, sneer won't do anything to it.

## Duplicate Observations

First, some preprocessing you _don't_ need to to do: some embedding methods ask
you to weed out duplicates in the data before proceeding, but `sneer` tries
to be robust enough to avoid that.

## Low Variance Column Filtering

However, if you have some columns with very low or zero variance (for instance,
because one column contains identical values for all observations), that can
cause some problems. Rather than require you to remove these ahead of time, 
these will be removed from the data frame automatically before other 
pre-processing takes place. Sneer will log to the console the number of columns
it filters in this way.

## The `scale_type` Parameter

By default, no scaling is applied to the input data. By providing an argument
to `scale_type` you can scale the data in the following way:

* `"auto"` Autoscales the data: each column is centered so that is has a mean of 
zero and then scaled so that its variance is 1.
* `"range"` Range scales each column of data: each column is scaled so that the 
values in each column have a range from 0 to 1.
* `"matrix"` Range scales the entire matrix: like range scaling, except the 
entire matrix is treated like one big column.

All these arguments can be abbreviated (e.g. to `"m"` instead of `"matrix"`).

Image data is often treated by range scaling the entire matrix, e.g. using the 
mnist data set as an example:

```res_mnist <- sneer(mnist, scale_type = "m")```

## What `sneer` Doesn't Do

Some data sets can be very high dimensional, and processing time can be
greatly reduced by doing PCA on the input data and keeping only a certain number
of the score vectors that result. 

Similarly, it's common to carry out various forms of whitening on image data 
sets. 

While this could become part of the preprocessing workflow, I've kept this out
of `sneer` for a couple of reasons. 

First, if you run an embedding with different parameters on the same dataset 
more than once, it's a bit of a waste of time to repeat PCA and/or whitening 
inside the internals of `sneer` rather than have you do it yourself once before 
doing any further embedding.

Second, there's often a subjective criteria involved in this form of 
preprocessing. Everyone has their own favorite way of doing it, so rather
than pointlessly expand the options even further, you should once again just
do it yourself once ahead of time.

At least that was a nice and gentle start to the option-wrangling. Having
preprocessed the input data, we can now move onto initializing the input data
that the embedding directly works on.

Previous: [Data Sets](datasets.html). Next: [Input Initialization](input-initialization.html). Up: [Index](index.html).

