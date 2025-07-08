# Chaos Game Sketch

This p5.js sketch generates fractal patterns via the chaos game.

### Overview

The Chaos Game is a simple algorithm that creates complex fractal patterns using randomness and geometric rules. 
Starting with a shape (like a triangle), you pick a random point and repeatedly move partway toward randomly chosen 
corners. Over many iterations, a pattern like the Sierpinski triangle emerges—order from chaos.

### How to Run

1. Run `npm install` to install the dependencies
2. Run `npm run dev` or `vite` to start the development server.
3. Open your browser and navigate to `http://localhost:5173/`.

### Controls

"Fade Out": When enabled, the canvas will fade gradually over time by layering a transparent background between frames.
When disabled, points accumulate permanently.

"Cycle Hues": When enabled, cycles the color hues of points as the simulation runs. When disabled, points are white.

"Show Pivots": Shows or hides the pivot points that the walker moves toward.

"Ruleset": Selects the algorithm used to determine the next pivot target for the walker.

"Interpolation": Chooses the interpolation strategy used to move between the current position and target pivot.

"Step Ratio": Determines how far the walker moves toward a chosen pivot on each step. (0.0 → 2.0)

"Number of Pivots": Sets how many pivot points define the shape (e.g., triangle, square, pentagon). (3 → 10)

"Zoom": Scales the distance of pivot points from the origin, effectively zooming in/out. (1/10 -> 10)

"Dot Size": Sets the rendered size of each plotted dot. (0.01 → 2.0)

"Animate": Toggles automatic cycling of the "Step Ratio," allowing for dynamic animation of the pattern.

"Animation Speed": Controls how quickly the animation modifies the "Step Ratio."

`Space`: Randomize the position of the pivots.

`ArrowUp` and `ArrowDown`: Scroll between different Chaos Game rulesets.

`<` and `>`: Scroll between different midpoint interpolation functions.

### To-Do

- [ ] Add more rulesets
- [ ] Add more shapes to map pivots onto (e.g. triangle, square, etc.)
- [ ] Currently, Zoom is more like a scale on the pivots. Add "dumb" zoom with transforms.
