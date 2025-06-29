import p5 from 'p5';

new p5((p: p5) => {
    const WIDTH = 1080;
    const HEIGHT = 720;

    const SPEED = 0.02;
    const HARMONIC_START = 1;
    const HARMONIC_SCALE = 2;

    const phasors: Phasor[] = [];

    let lastPoint: p5.Vector | null = null;
    let time = 0;
    let paused = true;

    p.setup = () => {
        p.createCanvas(WIDTH, HEIGHT);

        let omega = HARMONIC_START;
        for (let i = 0; i < 3; i++) {
            phasors.push({radius: 300 / (omega * Math.PI), phase: 0, omega: omega});
            omega += HARMONIC_SCALE;
        }

        p.stroke(255);
        p.background(0);
        p.noLoop();
    }

    p.draw = () => {
        // capture right side
        let rightSide = p.get(WIDTH / 2, 0, WIDTH, HEIGHT);
        p.background(0)
        p.image(rightSide, WIDTH / 2 + 1, 0)

        // Draw phasors
        p.push()
        p.translate(WIDTH / 4, HEIGHT / 2);
        p.strokeWeight(1);
        for (const phasor of phasors) {
            p.noFill();
            p.circle(0, 0, 2 * phasor.radius);

            p.rotate(phasor.omega * time - phasor.phase);
            p.line(0, 0, phasor.radius, 0);

            p.translate(phasor.radius, 0);
            p.fill(255);
            p.circle(0, 0, 6);

            p.rotate(-phasor.omega * time + phasor.phase);
        }
        p.pop()

        // Calculate the position of the phasor sum
        const xy = p.createVector(WIDTH / 4, HEIGHT / 2)
        for (const phasor of phasors) {
            xy.add(p5.Vector.fromAngle(phasor.omega * time - phasor.phase, phasor.radius));
        }

        // draw horizontal line representing the projection of the phasor sum
        p.strokeWeight(0.5);
        p.line(xy.x, xy.y, WIDTH / 2 + 1, xy.y);

        // Draw the traced function
        p.strokeWeight(2);
        if (lastPoint) {
            p.line(lastPoint.x + 1, lastPoint.y, WIDTH / 2 + 1, xy.y);
        } else {
            p.point(WIDTH / 2 + 1, xy.y);
        }
        lastPoint = p.createVector(WIDTH / 2 + 1, xy.y);

        // increment time
        time += SPEED;
    }

    p.mousePressed = (event: MouseEvent) => {
        console.log("mouseClicked:", event);
        if (event.button === 0) { // left click
            if (event.target.tagName === "CANVAS") {
                shuffle(phasors);
            }
        }
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

        if (event.code === "ArrowUp") {
            let omega = HARMONIC_START + HARMONIC_SCALE * phasors.length;
            phasors.push({radius: 300 / (omega * Math.PI), phase: 0, omega: omega});
        } else if (event.code === "ArrowDown") {
            phasors.pop();
        }

        if (phasors.length > 0) {
            if (event.code === "ArrowRight") {
                // shift the phasor list right
                const lastPhasor = phasors.pop();
                if (lastPhasor) phasors.unshift(lastPhasor);
            } else if (event.code === "ArrowLeft") {
                // shift the phasor list left
                const firstPhasor = phasors.shift()
                if (firstPhasor) phasors.push(firstPhasor);
            }
        }
    }

    function shuffle(array) {
        let currentIndex = array.length;

        // While there remain elements to shuffle...
        while (currentIndex != 0) {

            // Pick a remaining element...
            let randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex--;

            // And swap it with the current element.
            [array[currentIndex], array[randomIndex]] = [
                array[randomIndex], array[currentIndex]];
        }
    }

    // definitions
    type Phasor = {
        radius: number;
        phase: number;
        omega: number;
    };
});
