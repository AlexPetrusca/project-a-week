# Chaos Game Sketch

This p5.js sketch generates fractal patterns via the chaos game.

### Overview

The Chaos Game is a simple algorithm that creates complex fractal patterns using randomness and geometric rules. 
Starting with a shape (like a triangle), you pick a random point and repeatedly move partway toward randomly chosen 
corners. Over many iterations, a pattern like the Sierpinski triangle emerges—order from chaos. Despite the randomness,
the result is a highly structured and self-similar fractal.

### To-Do

- [ ] Add more rulesets
- [ ] Add more shapes to map pivots onto (e.g. triangle, square, etc.)
- [ ] Currently, Zoom is more like a scale on the pivots. Add "dumb" zoom with transforms.
