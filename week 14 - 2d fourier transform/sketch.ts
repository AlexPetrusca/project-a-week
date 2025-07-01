import p5 from "p5";
import sketchXY from './sketch-xy';
import sketchComplex from './sketch-complex';

const SKETCHES = [sketchXY, sketchComplex];
let currentSketch: p5 | null = null;
let currentSketchId: number = 0;

function injectSketch(sketchFn: (p: p5) => void) {
    if (currentSketch) {
        currentSketch.remove(); // clean up existing sketch
    }
    currentSketch = new p5(sketchFn, document.getElementById('sketch-container') as HTMLElement);
}

function updateSketch() {
    injectSketch(SKETCHES[currentSketchId % SKETCHES.length]);
}

window.addEventListener('keydown', e => {
   if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
       if (e.key === 'ArrowLeft') {
           currentSketchId--;
       } else {
           currentSketchId++;
       }
       updateSketch();
   }
});

updateSketch();
