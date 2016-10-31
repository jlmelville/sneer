---
title: "Embedding Concepts"
output: html_document
---

Next: [Data Sets](datasets.html). Up: [Index](index.html).

Let's get some jargon out of the way so I can mention it later without
having to painstakingly redefine what I mean every time.

When I talk about an **embedding**, I mean any of the methods that `sneer`
can use to generate lower-dimensional (normally 2D) coordinates from input 
data. The set of output coordinates is the output **configuration**.

These methods have a cost function which measures how the output configuration
compares to the input configuration. `sneer` attempts to optimize the 
output configuration to minimize the cost function.

Traditional embedding methods like metric MDS or Sammon mapping have cost 
functions where the distances in the input and output spaces appear directly:
give or take a weighting, they are attempting to match the distances in the
input and output space. I'll call these **distance-based** embedding methods.

But if you look at the cost function of t-SNE, you won't see any distances.
Instead you see probabilities. These are related to the distances via a series
of transformations, but it's not a direct connection and this seems to have a
significant effect on the results. I'll call these **probability-based** 
embedding methods.

The process of generating probabilities from distances goes like this:

* Create a weight matrix from the distances. The weights are generally such that 
  the bigger the weight, the shorter the distances, so you can also think of
  them as similarities. I'll call the function that generates the weights from 
  the distances the **kernel function** or the **weight function**.
  
* The weights are then normalized into probabilities. For more about the 
  normalizations that are often used, see the [gradients](gradients.html) page.

Next: [Data Sets](datasets.html). Up: [Index](index.html).


