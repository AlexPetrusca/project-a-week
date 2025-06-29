import p5 from 'p5';
import coords from './assets/ellipse.json';

new p5((p: p5) => {
    const WIDTH = 800;
    const HEIGHT = 800;

    const SPEED = 0.02;

    let timeSignal: Signal2D;
    let freqSignal: FreqSignal2D;
    const phasors: Phasor[] = [];

    let time = 0;
    let paused = true;

    p.setup = () => {
        p.createCanvas(WIDTH, HEIGHT);
        p.stroke(255);
        p.background(0);
        p.noLoop();

        const aspectRatio = normalizeCoordinates(coords);
        console.log("aspectRatio:", aspectRatio);

        const coords2 = [];
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 10; j++) {
                coords2.push(100, 0);
            }
            for (let j = 0; j < 10; j++) {
                coords2.push(-100, 0);
            }
        }

        timeSignal = createSignal2D(coords2);
        const fxs = dft(timeSignal.xs);
        const fys = dft(timeSignal.ys);
        freqSignal = createFreqSignal2D(fxs, fys);

        console.log("timeSignal:", timeSignal);
        console.log("freqSignal:", freqSignal);

        const N = freqSignal.length;
        const BIN_WIDTH = N;
        for (let i = 0; i < N; i++) {
            if (i === 0 || i === N / 2) {
                phasors.push({
                    radius: freqSignal.fxs[i].magnitude(),
                    phase: freqSignal.fxs[i].iangle(),
                    omega: BIN_WIDTH * i / N,
                })
            } else if (i < N / 2) {
                phasors.push({
                    radius: freqSignal.fxs[i].magnitude(),
                    phase: freqSignal.fxs[i].iangle(),
                    omega: BIN_WIDTH * i / N,
                })
                phasors.push({
                    radius: freqSignal.fxs[N - i].magnitude(),
                    phase: freqSignal.fxs[N - i].iangle(),
                    omega: -BIN_WIDTH * (i) / N,
                })
            }
        }
        console.log("phasors:", phasors);
    }

    let lastPoint: p5.Vector | null = null; // todo: DELETE ME
    p.draw = () => {
        // let scale = WIDTH / 2 - 10;
        // let xOffset = WIDTH / 2;
        // let yOffset = HEIGHT / 2;
        //
        // // Plot coordinates
        // p.strokeWeight(2);
        // for (let i = 0; i < timeSignal.length; i++) {
        //     const x = scale * timeSignal.xs[i] + xOffset;
        //     const y = scale * timeSignal.ys[i]  + yOffset;
        //     p.point(x, y);
        // }
        //
        // // Bounding box
        // p.stroke(255, 255, 255, 100);
        // p.noFill();
        // p.strokeWeight(1);
        // p.rect(xOffset, yOffset, scale, scale);
        //
        // // Increment time
        // time += SPEED;




        // capture right side
        let rightSide = p.get(WIDTH / 2, 0, WIDTH, HEIGHT);
        p.background(0)
        p.image(rightSide, WIDTH / 2 + 1, 0)

        // Draw phasors
        p.push()
        p.translate(WIDTH / 4, HEIGHT / 2);
        p.strokeWeight(1);
        for (const phasor of phasors) {
            const phi = phasor.omega * time - phasor.phase;

            p.noFill();
            p.circle(0, 0, 2 * phasor.radius);

            p.rotate(phi);
            p.line(0, 0, phasor.radius, 0);

            p.translate(phasor.radius, 0);
            p.fill(255);
            p.circle(0, 0, 6);

            p.rotate(-phi);
        }
        p.pop()

        // Calculate the position of the phasor sum
        const xy = p.createVector(WIDTH / 4, HEIGHT / 2)
        for (const phasor of phasors) {
            const phi = phasor.omega * time - phasor.phase;
            xy.add(p5.Vector.fromAngle(phi, phasor.radius));
        }

        // Draw horizontal line representing the projection of the phasor sum
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
    }

    function dft(timeSignal: number[]) {
        const freqSignal: Complex[] = [];

        const N = timeSignal.length;
        for (let k = 0; k < N; k++) {
            let component = new Complex(0, 0);
            for (let n = 0; n < N; n++) {
                const phi = 2 * Math.PI * k * n / N;
                const re = timeSignal[n] * Math.cos(phi);
                const im = -timeSignal[n] * Math.sin(phi);
                component = component.add(new Complex(re, im));
            }
            freqSignal.push(component.scale(1/N));
        }

        return freqSignal;
    }

    function createFreqSignal2D(fxs: Complex[], fys: Complex[]): FreqSignal2D {
        return {
            fxs: fxs,
            fys: fys,
            length: fxs.length,
        };
    }

    function createSignal2D(xys: number[]): Signal2D {
        return {
            xs: xys.filter((_, i) => i % 2 === 0),
            ys: xys.filter((_, i) => i % 2 === 1),
            length: xys.length / 2,
        };
    }

    function normalizeCoordinates(coords: number[]): number {
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
        return aspectRatio;
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

    type Signal2D = {
        xs: number[];
        ys: number[];
        length: number;
    }

    type FreqSignal2D = {
        fxs: Complex[];
        fys: Complex[];
        length: number;
    }

    class Complex {
        re: number;
        im: number;

        constructor(re: number, im: number) {
            this.im = im;
            this.re = re;
        }

        static fromPolar(r: number, theta: number): Complex {
            return new Complex(r * Math.cos(theta), r * Math.sin(theta));
        }

        add(other: Complex): Complex {
            return new Complex(this.re + other.re, this.im + other.im);
        }

        mult(other: Complex): Complex {
            return new Complex(
                this.re * other.re - this.im * other.im,
                this.re * other.im + this.im * other.re
            );
        }

        scale(scalar: number): Complex {
            return new Complex(this.re * scalar, this.im * scalar);
        }

        magnitude(): number {
            return Math.sqrt(this.re ** 2 + this.im ** 2);
        }

        angle(): number {
            return Math.atan2(this.im, this.re);
        }

        iangle(): number {
            return Math.atan2(this.re, this.im);
        }
    }
});
