An **integral image** or *summed-area table* is a data structure for quickly and efficiently generating the sum of values in a rectangular subset of a grid.

![[integral_image_overview.png]]

An integral image can be thought of as a 2-dimensional extension of a [[prefix sum]]. The method underlying integral images can be easily extended to higher dimensional images.

### Operation

The value for a pixel in an **integral image** is the sum of the pixels _above and to the left_ of the pixel in the **source image**.
- Bottom-right pixel in integral image (value = 113), represents the sum of all the pixels in the source image.

![[integral_image_algorithm.jpg]]

An integral image allows us to perform quick arithmetic operations over chunks of a source image.
- One time $O(n)$ cost for all pixels —> then, $O(1)$ arithmetic operations over pixel chunks afterwords
- Used in Haar Cascade face detection
- Useful in interview problems involving matrices. (Its an idea worth throwing at the problem to optimize it)

![[integral_image_compute_sum.jpg]]

### Applications

Integral images were originally invented for use with mipmaps (pre-calculated sequences of images, each of which is a progressively lower resolution representation of the previous). 

Today, they are also used extensively for image convolution operations in domains like computer vision and machine learning. Integral images were prominently used within the Viola–Jones object detection framework in 2001.