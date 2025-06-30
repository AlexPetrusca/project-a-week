import p5 from 'p5';
import ellipseCoords from './assets/ellipse.json';
import spiralCoords from './assets/spiral.json';
import flagCoords from './assets/flag.json';
import squareWaveCoords from './assets/square-wave.json';
import trainCoords from './assets/train.json';
import leafCoords from './assets/leaf.json';
import handCoords from './assets/hand.json';
import womanCoords from './assets/woman.json';
import dogCoords from './assets/dog.json';
import dinosaurCoords from './assets/dinosaur.json';
import birdCoords from './assets/bird.json';
import sexyCoords from './assets/sexy.json';
import indianCoords from './assets/indian.json';
import catCoords from './assets/cat.json';

new p5((p: p5) => {
    const WIDTH = 1080;
    const HEIGHT = 720;

    const SCALE = 150;
    const BIN_WIDTH = 1;
    const SPEED = 0.02;
    const COORDS_MAP = new Map([
        ['Ellipse', ellipseCoords],
        ['Spiral', spiralCoords],
        ['Flag', flagCoords],
        ['Train', trainCoords],
        ['Leaf', leafCoords],
        ['Hand', handCoords],
        ['Cat', catCoords],
        ['Dog', dogCoords],
        ['Bird', birdCoords],
        ['Dinosaur', dinosaurCoords],
        ['Indian', indianCoords],
        ['Woman', womanCoords],
        ['Sexy', sexyCoords],
        ['Square Wave', squareWaveCoords],
    ]);

    let coords: number[] = leafCoords;
    let timeSignal: Signal2D;
    let freqSignal: FreqSignal2D;
    const phasorsX: Phasor[] = [];
    const phasorsY: Phasor[] = [];

    let time = -SPEED;
    const trace: p5.Vector[] = [];

    let quality = 1;
    let isPaused = true;
    let isTwoSided = true;

    p.setup = () => {
        p.createCanvas(WIDTH, HEIGHT);
        p.background(0);
        p.noLoop();

        updateSignal();
        updatePhasors();
        initUI();
    }

    p.draw = () => {
        if (!isPaused) {
            time += SPEED;
        }

        p.background(0);



        let margin = 20;
        let scale = (HEIGHT - margin) / 2;
        let yOffset = scale;
        let xOffset = WIDTH - scale - margin;

        // Plot coordinates
        p.stroke(255, 100);
        p.strokeWeight(2);
        for (let i = 0; i < timeSignal.length; i++) {
            const x = 0.5 * scale * (timeSignal.xs[i] + 1) + xOffset;
            const y = 0.5 * scale * (timeSignal.ys[i] + 1) + yOffset;

            if (i === 0 || i == timeSignal.length - 1) {
                p.strokeWeight(8);
            } else {
                p.strokeWeight(4);
            }

            if (i % 3 === 0) {
                p.stroke(255, 0, 0);
            } else if (i % 3 === 1) {
                p.stroke(0, 255, 0);
            } else {
                p.stroke(0, 0, 255)
            }

            p.point(x, y);
        }

        for (let i = 0; i < timeSignal.length - 1; i++) {
            const x1 = 0.5 * scale * (timeSignal.xs[i] + 1) + xOffset;
            const y1 = 0.5 * scale * (timeSignal.ys[i] + 1) + yOffset;
            const x2 = 0.5 * scale * (timeSignal.xs[i + 1] + 1) + xOffset;
            const y2 = 0.5 * scale * (timeSignal.ys[i + 1] + 1) + yOffset;
            p.strokeWeight(1);
            p.stroke(255)
            p.line(x1, y1, x2, y2);
        }

        // Bounding box
        p.stroke(255, 100);
        p.strokeWeight(1);
        p.noFill();
        p.rect(xOffset, yOffset, scale, scale);



        // Calculate the position of the phasor sum
        let startVectorX = p.createVector(0.5 * WIDTH, 0.25 * HEIGHT);
        let startVectorY = p.createVector(0.25 * HEIGHT, 0.75 * HEIGHT - margin);
        const numPhasors = Math.floor(quality * phasorsX.length);
        const posX = calculatePhasorPosition(phasorsX, numPhasors, startVectorX.copy());
        const posY = calculatePhasorPosition(phasorsY, numPhasors, startVectorY.copy());

        // Build and manage the trace of the phasor sum
        const TRACE_LIMIT = 500;
        if (!isPaused) {
            trace.unshift(p.createVector(posX.x, posY.y));
            if (trace.length > TRACE_LIMIT) {
                trace.pop();
            }
        }

        // Draw the trace
        p.strokeWeight(2);
        for (let i = trace.length - 1; i > 0; i--) {
            p.stroke(Math.floor(255 * ((TRACE_LIMIT - i) / TRACE_LIMIT)));
            p.line(trace[i].x, trace[i].y, trace[i - 1].x, trace[i - 1].y);
        }

        // Draw phasors for x-axis
        p.stroke(255);
        p.strokeWeight(1);
        p.push()
        p.translate(startVectorX);
        for (let i = 0; i < numPhasors; i++) {
            const phasor = phasorsX[i];
            const phi = phasor.omega * time - phasor.phase;

            p.noFill();
            p.circle(0, 0, 2 * phasor.radius);

            p.rotate(phi);
            p.line(0, 0, phasor.radius, 0);

            p.translate(phasor.radius, 0);
            p.fill(255);
            p.circle(0, 0, 5);

            p.rotate(-phi);
        }
        p.pop()

        // Draw phasors for y-axis
        p.stroke(255);
        p.strokeWeight(1);
        p.push()
        p.translate(startVectorY);
        for (let i = 0; i < numPhasors; i++) {
            const phasor = phasorsY[i];
            const phi = phasor.omega * time - phasor.phase;

            p.noFill();
            p.circle(0, 0, 2 * phasor.radius);

            p.rotate(phi);
            p.line(0, 0, phasor.radius, 0);

            p.translate(phasor.radius, 0);
            p.fill(255);
            p.circle(0, 0, 5);

            p.rotate(-phi);
        }
        p.pop()

        // Draw the projections of the x-axis and y-axis phasors
        p.strokeWeight(0.5);
        p.line(posX.x, posX.y, posX.x, posY.y);
        p.line(posY.x, posY.y, posX.x, posY.y);
    }

    function calculatePhasorPosition(phasors: Phasor[], numPhasors: number, startVector: p5.Vector): p5.Vector {
        const xy = startVector;
        for (let i = 0; i < numPhasors; i++) {
            const phasor = phasors[i];
            const phi = phasor.omega * time - phasor.phase;
            xy.add(p5.Vector.fromAngle(phi, phasor.radius));
        }
        return xy;
    }

    p.mousePressed = (event: MouseEvent) => {
        console.log("mouseClicked:", event);
        if (event.button === 0) { // left click
            const target = event.target as HTMLElement;
            if (target.tagName === "CANVAS") {
                shuffle(phasorsX);
                shuffle(phasorsY);
            }
        }
        p.draw();
    }

    p.keyPressed = (event: KeyboardEvent) => {
        console.log("keyPressed:", event);
        if (event.code === "Space") {
            isPaused = !isPaused;
            if (isPaused) {
                p.noLoop();
            } else {
                p.loop();
            }
        }
        if (event.code === "ArrowUp") {
            isTwoSided = !isTwoSided;
            updatePhasors();
        }
        p.draw();
    }

    function initUI() {
        const sel: P5Select = p.createSelect() as P5Select;
        sel.position(10, 10);
        for (const key of COORDS_MAP.keys()) {
            sel.option(key);
        }
        sel.selected('Leaf');
        sel.changed(() => {
            coords = COORDS_MAP.get(sel.value()) as number[];
            updateSignal();
            updatePhasors();
            p.draw();
        });

        const sli: P5Slider = p.createSlider(0, 1, 1, 0.001) as P5Slider;
        sli.position(10, 35);
        sli.input(() => {
            quality = sli.value() as number;
            p.draw();
        });
    }

    function updateSignal() {
        const aspectRatio = normalizeCoordinates(coords);
        console.log("aspectRatio:", aspectRatio);

        timeSignal = createSignal2D(coords);
        const fxs = dft(timeSignal.xs);
        const fys = dft(timeSignal.ys);
        freqSignal = createFreqSignal2D(fxs, fys);

        console.log("timeSignal:", timeSignal);
        console.log("freqSignal:", freqSignal);
    }

    function updatePhasors() {
        // clear phasors
        phasorsX.length = 0;
        phasorsY.length = 0;

        if (isTwoSided) {
            twoSidedPhasors(SCALE, BIN_WIDTH);
        } else {
            oneSidedPhasors(SCALE, BIN_WIDTH);
        }
        console.log("phasorsX:", phasorsX);
        console.log("phasorsY:", phasorsY);
    }

    function twoSidedPhasors(scale: number, binWidth: number) {
        function apply(freqSignal1D: Complex[], phasors: Phasor[], rotation: number = 0) {
            const N = freqSignal1D.length;
            for (let i = 0; i < N; i++) {
                if (i === 0 || i === N / 2) { // DC and Nyquist component
                    phasors.push({
                        radius: scale * freqSignal1D[i].magnitude(),
                        phase: freqSignal1D[i].angle() - rotation,
                        omega: binWidth * i,
                    });
                } else if (i < N / 2) {
                    // positive frequency component
                    phasors.push({
                        radius: scale * freqSignal1D[i].magnitude(),
                        phase: freqSignal1D[i].angle() - rotation,
                        omega: binWidth * i,
                    });
                    // negative frequency component
                    phasors.push({
                        radius: scale * freqSignal1D[N - i].magnitude(),
                        phase: freqSignal1D[N - i].angle() - rotation,
                        omega: -binWidth * i,
                    });
                }
            }
        }

        apply(freqSignal.fxs, phasorsX, 0);
        apply(freqSignal.fys, phasorsY, Math.PI / 2);
    }

    function oneSidedPhasors(scale: number, binWidth: number) {
        function apply(freqSignal1D: Complex[], phasors: Phasor[], rotation: number = 0) {
            const N = freqSignal1D.length;
            for (let i = 0; i < N; i++) {
                if (i === 0 || i === N / 2) { // DC and Nyquist component
                    phasors.push({
                        radius: scale * freqSignal1D[i].magnitude(),
                        phase: freqSignal1D[i].angle() - rotation,
                        omega: binWidth * i,
                    });
                } else if (i < N / 2) {
                    // sum of positive and negative components (mirrored)
                    phasors.push({
                        radius: 2 * scale * freqSignal1D[i].magnitude(),
                        phase: freqSignal1D[i].angle() - rotation,
                        omega: binWidth * i,
                    });
                }
            }
        }

        apply(freqSignal.fxs, phasorsX, 0);
        apply(freqSignal.fys, phasorsY, Math.PI / 2);
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
                const yLimit = 1 / aspectRatio;
                coords[i] = p.map(coords[i], minX, maxX, -1, 1);
                coords[i + 1] = p.map(coords[i + 1], minY, maxY, -yLimit, yLimit);
            }
        } else {
            for (let i = 0; i < coords.length; i += 2) {
                const xLimit = aspectRatio;
                coords[i] = p.map(coords[i], minX, maxX, -xLimit, xLimit);
                coords[i + 1] = p.map(coords[i + 1], minY, maxY, -1, 1);
            }
        }
        return aspectRatio;
    }

    function shuffle(array: any[]) {
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

    // missing types definitions
    type P5Select = {
        selected(val?: string): string | void;
        changed(cb: () => void): void;
        option(name: string, value?: string): void;
        value(): string;
    } & p5.Element;

    type P5Slider = {
        input(cb: () => void): void;
        changed(cb: () => void): void;
        value(): number;
    } & p5.Element;
});

