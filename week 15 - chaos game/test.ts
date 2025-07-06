import p5 from 'p5';
// @ts-ignore
import vertexShader from './shader.vert?raw';
// @ts-ignore
import fragmentShader from './shader.frag?raw';

new p5((p: p5) => {
    let pg: p5.Graphics;

    p.setup = () => {
        p.createCanvas(400, 400);
        pg = p.createGraphics(400, 400, p.WEBGL);
        pg.noStroke();
        pg.background(0);
    }

    p.draw = () => {
        // Apply decay: draw a translucent black rectangle over everything
        pg.push();
        pg.resetShader(); // Use default shader for decay pass
        pg.fill(0, 5);   // Low alpha → slow fade, higher alpha → faster
        pg.rectMode(p.CENTER);
        pg.rect(0, 0, p.width, p.height);
        pg.pop();

        // Draw new white content — this will slowly fade out
        pg.push();
        pg.fill(255);
        pg.ellipse(p.random(-200, 200), p.random(-200, 200), 10, 10);
        pg.pop();

        // Draw buffer to main canvas
        p.image(pg, 0, 0);
    }
});
