# Fourier Series Sketch

This p5.js sketch demonstrates how to use the Fourier series and phasors to approximate a square wave. It can be easily
adapted to visualize other periodic functions as well.

### Overview

The **Fourier series** is a mathematical tool used to represent *periodic functions* as an *infinite sum of 
trigonometric functions* (i.e. sine and cosine functions). By taking more and more terms, we get increasingly accurate
approximations of the original function.

### How to Run

1. Run `npm install` to install the dependencies
2. Run `npm run dev` or `vite` to start the development server. 
3. Open your browser and navigate to `http://localhost:5173/`.

### Controls

`Space`: Toggle the animation on and off.

`ArrowUp` or `ArrowDown`: Increase or decrease the number of phasors.

`ArrowRight` or `ArrowLeft`: Shift the order of the phasors left or right.

`MouseClick`: Randomize the order of the phasors.
