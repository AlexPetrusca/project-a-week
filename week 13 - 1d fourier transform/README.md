# 1D Fourier Transform Sketch

This p5.js sketch demonstrates how the discrete Fourier transform (DFT) decomposes a signal into a sum of phasors.

### Overview

The discrete Fourier transform translates between two different ways to represent a signal:
- The **time domain** representation (a series of evenly spaced samples over time)
- The **frequency domain** representation (the strength and phase of waves, at different frequencies, that can be used 
  to reconstruct the signal)

Given $N$ evenly spaced samples, the DFT outputs $N$ complex numbers representing phasors at different frequencies. 
Summing these phasors and "driving" them at their respective frequencies reconstructs the original signal.

### How to Run

1. Run `npm install` to install the dependencies
2. Run `npm run dev` or `vite` to start the development server.
3. Open your browser and navigate to `http://localhost:5173/`.

### Controls

The dropdown selects the signal.

The slider controls the quality of the recreation (i.e., number of phasors used).

`Space`: Toggle the animation on and off.

`ArrowUp`: Toggle between a 2-sided and 1-sided DFT (Default: 1-sided).

`ArrowDown`: Toggle between the x and y component of the signal (Default: y component).

`MouseClick`: Randomize the order of the phasors.

### Credits

https://shinao.github.io/PathToPoints/ - Used to convert SVG files into point-based JSON representations.

