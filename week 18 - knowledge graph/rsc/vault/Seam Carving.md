**Seam carving** is a content-aware image resizing technique that allows for the removal or addition of pixels in an image while preserving its important visual content.

![[algo_seam_carving_overview.png]]

Unlike traditional resizing methods, such as cropping or scaling, which can distort or remove key features, seam carving intelligently identifies and manipulates "seams"—paths of least importance—through the image.

### How It Works

##### Energy Map Creation
    
The algorithm calculates an "energy function" for each pixel, which measures its importance. Common metrics include gradient magnitude (detecting edges) or other saliency measures that highlight areas of high contrast or detail, like faces or objects, as more significant.

![[algo_seam_carving_energy_map.png]] { caption="Example of using a vertical and horizontal sobel filter (edge detector) to extract an energy map." }

##### Seam Identification
    
A seam is a connected path of pixels (either vertical or horizontal) from one edge of the image to the opposite edge. The algorithm finds the seam with the lowest total energy, meaning it passes through the least important regions of the image.

![[algo_seam_carving_seam_identification.png]] { caption="Left: the original image, Right: a visualization of the cumulative energy per pixel calculated along all paths starting from that pixel." }

Seam carving typically uses dynamic programming to efficiently compute the optimal seam by minimizing the cumulative energy along the path.

![[algo_seam_carving_compute_seam.png]] { caption="Example of converting from energy to cumulative energy, Left: the cumulative energy representation is computed bottom-up from the energy representation using a dynamic programming approach (here we only consider the optimal path, but this happens for all paths), Right: we can follow the minimal cumulative energy down to get the seam with the smallest total energy." }
    
##### Seam Removal or Insertion

- For shrinking an image, the lowest-energy seam is removed, and the image is stitched back together.
- For expanding an image, seams can be duplicated or new pixels inserted along low-energy paths.
- This process repeats iteratively until the desired size is achieved.

![[algo_seam_carving_demo.gif]] { caption="Example of shrinking an image using iterative seam removal." }

### Pros / Cons

`~plus_bullet` Preserves important content better than cropping or uniform scaling.
`~plus_bullet` Adapts to the image's context, making it versatile for photos with complex layouts.
`~minus_bullet` Works best on images with clear distinctions between important and unimportant areas (e.g., landscapes with sky or plain backgrounds).
`~minus_bullet` Can produce artifacts or distortions if applied excessively or on images with uniform detail (e.g., a busy crowd).
`~minus_bullet` Computationally intensive compared to simple resizing, especially for large images or videos.

### Applications

- **Image Resizing**: Adjusts images to fit different aspect ratios without distorting key features.
- **Object Removal**: By marking specific areas as high-energy (to preserve) or low-energy (to remove), users can eliminate unwanted objects.
- **Content-Aware Scaling**: Used in software like Adobe Photoshop (e.g., its "Content-Aware Scale" tool).