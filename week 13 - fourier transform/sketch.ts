import p5 from 'p5';
import coords from './assets/ellipse.json'; // This import style requires "esModuleInterop", see "side notes"

new p5((p: p5) => {
    const WIDTH = 800;
    const HEIGHT = 800;

    const SPEED = 0.02;

    const phasors: Phasor[] = [];

    let lastPoint: p5.Vector | null = null;
    let time = 0;
    let paused = true;

    p.setup = () => {
        p.createCanvas(WIDTH, HEIGHT);

        // normalize coordinates
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        for (let i = 0; i < coords.length; i += 2) {
            const x = coords[i];
            const y = coords[i + 1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }
        const width = maxX - minX;
        const height = maxY - minY;
        const aspectRatio = width / height;
        console.log("aspectRatio:", aspectRatio);
        if (aspectRatio > 1) {
            for (let i = 0; i < coords.length; i += 2) {
                coords[i] = p.map(coords[i], minX, maxX, 0, 1);
                coords[i + 1] = p.map(coords[i + 1], minY, maxY, 0, 1 / aspectRatio);
            }
        } else {
            for (let i = 0; i < coords.length; i += 2) {
                coords[i] = p.map(coords[i], minX, maxX, 0, aspectRatio);
                coords[i + 1] = p.map(coords[i + 1], minY, maxY, 0, 1);
            }
        }

        p.stroke(255);
        p.background(0);
        p.noLoop();
    }

    p.draw = () => {
        let scale = WIDTH / 2 - 10;
        let xOffset = WIDTH / 2;
        let yOffset = HEIGHT / 2;

        p.strokeWeight(2);
        for (let i = 0; i < coords.length; i += 2) {
            const x = scale * coords[i] + xOffset;
            const y = scale * coords[i + 1]  + yOffset;
            p.point(x, y);
        }

        p.stroke(255, 255, 255, 100);
        p.noFill();
        p.strokeWeight(1);
        p.rect(xOffset, yOffset, scale, scale);

        // // capture right side
        // let rightSide = p.get(WIDTH / 2, 0, WIDTH, HEIGHT);
        // p.background(0)
        // p.image(rightSide, WIDTH / 2 + 1, 0)
        //
        // // Draw phasors
        // p.push()
        // p.translate(WIDTH / 4, HEIGHT / 2);
        // p.strokeWeight(1);
        // for (const phasor of phasors) {
        //     p.noFill();
        //     p.circle(0, 0, 2 * phasor.radius);
        //
        //     p.rotate(phasor.omega * time - phasor.phase);
        //     p.line(0, 0, phasor.radius, 0);
        //
        //     p.translate(phasor.radius, 0);
        //     p.fill(255);
        //     p.circle(0, 0, 6);
        //
        //     p.rotate(-phasor.omega * time + phasor.phase);
        // }
        // p.pop()
        //
        // // Calculate the position of the phasor sum
        // const xy = p.createVector(WIDTH / 4, HEIGHT / 2)
        // for (const phasor of phasors) {
        //     xy.add(p5.Vector.fromAngle(phasor.omega * time - phasor.phase, phasor.radius));
        // }
        //
        // // draw horizontal line representing the projection of the phasor sum
        // p.strokeWeight(0.5);
        // p.line(xy.x, xy.y, WIDTH / 2 + 1, xy.y);
        //
        // // Draw the traced function
        // p.strokeWeight(2);
        // if (lastPoint) {
        //     p.line(lastPoint.x + 1, lastPoint.y, WIDTH / 2 + 1, xy.y);
        // } else {
        //     p.point(WIDTH / 2 + 1, xy.y);
        // }
        // lastPoint = p.createVector(WIDTH / 2 + 1, xy.y);
        //
        // // increment time
        // time += SPEED;
    }

    // @ts-ignore
    p.mousePressed = (event: MouseEvent) => {
    }

    p.keyPressed = (event: KeyboardEvent) => {
        console.log("keyPressed:", event);
        if (event.code === "Space") {
            paused = !paused;
            if (paused) {
                p.noLoop();
            } else {
                p.loop();
            }
        }
    }

    // definitions & exports
    type Phasor = {
        radius: number;
        phase: number;
        omega: number;
    };
});
