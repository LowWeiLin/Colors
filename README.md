# Colors

Sometimes, Life is too colorful, filled with different shades and hues. <br>
Wouldn't it be nice if we can simplify everything?

Colors! Enables you to: <br>

1. Quantize the colors in an image (find dominant colors in an image)<br>
2. Identify the names of the colors (because who knows that #808080 is called "Fractal")<br>

## What?

![alt tag](./Documentation/images/balls.png)

Optimal K = 4 <br>
[ 39.1492537313 % ] Gray , SteelBlue , SteelBlue - (69, 129, 176) <br>
[ 20.6417910448 % ] Orange , GoldenRod , Goldenrod2 - (231, 184, 14) <br>
[ 20.2835820896 % ] Olive , OliveDrab , OliveDrab4 - (76, 124, 18) <br>
[ 19.9253731343 % ] Red , Crimson , Red3 - (221, 14, 15) <br>

## How?

1. Color quantization is done via k-means clustering.<br>
   The OpenCV documents provide a good explaination of k-means, as well on applying it on images.
   What is not so clear, is how to select a number K, the number of clusters for k-means to find.<br>
   For this code, the compactness value returned by the k-means algorithm is analyzed, which usually gives a nice slope.
   The elbow method is then used to determine the optimal K to be used.
   Although it works well, the method and code used here are purely experimental,
   and more info should be sought about [determining the number of clusters in a dataset](http://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set).<br>
   Another caveat is that the optimal K that can be identified with this method is >= 3. If the 'actual' optimal K is 1 or 2, this method would not work.

2. Color identification/naming is done by comparing from a known list of rgb-name maps.<br>
   The code is mostly from [here](https://gist.github.com/jdiscar/9144764) and quite self explanatory.<br>
   Some modifications were done to add another color list, and to use cielab color space for comparison between two colors,
   using conversion code from [here](http://www.cse.unr.edu/~quiroz/index.php?option=code).

## Info

Colors is written in python, using OpenCV and other modules.
It is inspired by: http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
Code has been taken/adapted from multiple sources, documented in the code.
