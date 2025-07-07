import p5 from 'p5';
// @ts-ignore
import fadeVertShader from './shader.vert?raw';
// @ts-ignore
import fadeFragShader from './shader.frag?raw';

new p5((p: p5) => {
    let NUM_PIVOTS = 5; // number of pivots
    let STEP_RATIO = 0.5; // how far to step towards the pivot
    const NUM_ITERATIONS = 5000; // number of iterations for the chaos walk
    const NUM_PAGES = 4; // number of different chaos walk methods

    let pivots: p5.Vector[] = [];
    let chaosPoint: p5.Vector;
    let hueOffset = 0;

    let lastIdx: number = 0;
    let secondLastIdx: number = 0;

    let pgMain: p5.Graphics;
    let pgFade: p5.Graphics;
    let fadeShader: p5.Shader;

    let accumulate = true; // whether to accumulate points or not
    let page = 0; // current page for different chaos walk methods
    let lerpFn: LerpFunction = basicLerp; // default lerp function
    let lerpId = 0; // id of the lerp function
    let lerpFnMap: Map<number, LerpFunction> = new Map([
        [0, basicLerp],
        [1, bezierLerp],
        [2, noisyBezierLerp],
        [3, cubicBezierLerp],
        [4, arcLerp],
        [5, catmullRomLerp],
        [6, perlinLerp],
        [7, sinusoidalLerp],
        [8, magnetLerp],
        [9, perlinMagnetLerp],
        [10, experimentLerp],
    ]);

    p.setup = (width: number = 1080, height: number = 720) => {
        p.createCanvas(width, height, p.WEBGL);
        p.colorMode(p.HSB, 360, 100, 100, 100)

        // Render the scene and fade it in off-screen buffers
        pgMain = p.createGraphics(width, height, p.WEBGL);
        pgFade = p.createGraphics(width, height, p.WEBGL);
        fadeShader = pgFade.createShader(fadeVertShader, fadeFragShader);
        pgFade.shader(fadeShader);

        initScene();
        initUI();
    }

    function initUI() {
        let line1 = p.createDiv();
        p.createSpan("Step Ratio:").parent(line1);
        const stepRatioSlider: P5Slider = p.createSlider(0, 2, STEP_RATIO, 0.005).parent(line1) as P5Slider;
        stepRatioSlider.input(() => {
            STEP_RATIO = stepRatioSlider.value();
            pgMain.background(0);
        });

        let line2 = p.createDiv();
        p.createSpan("Number of Pivots:").parent(line2);
        const numPivotsSlider: P5Slider = p.createSlider(3, 10, NUM_PIVOTS, 1).parent(line2) as P5Slider;
        numPivotsSlider.input(() => {
            NUM_PIVOTS = numPivotsSlider.value();
            initScene();
        });
    }

    function initScene() {
        pivots = [];

        let phase = 2 * Math.PI * Math.random();
        const radius = (p.height - 100)/2;
        for (let i = 0; i < NUM_PIVOTS; i++) {
            const x = radius * Math.cos(phase);
            const y = radius * Math.sin(phase);
            pivots.push(p.createVector(x, y));
            phase += 2 * Math.PI / NUM_PIVOTS; // spread pivots evenly around a circle
        }

        chaosPoint = p.createVector(p.random(p.width), p.random(p.height));
        pgMain.background(0);
    }

    p.draw = () => {
        drawPivots();
        drawChaosWalk();
        if (!accumulate) {
            applyGlobalFade();
        } else {
            p.image(pgMain, -p.width / 2, -p.height / 2);
        }

        // STEP_RATIO += 0.0005; // slowly increase the step ratio to create a dynamic effect
    }

    function drawPivots() {
        pgMain.stroke(255);
        pgMain.strokeWeight(8);
        for (const pivot of pivots) {
            pgMain.point(pivot.x, pivot.y);
        }
    }

    function applyGlobalFade() {
        // fadeShader.setUniform('u_decay', 0.95); // decay factor
        fadeShader.setUniform('u_texture', pgMain);
        pgFade.rect(-p.width / 2, -p.height / 2, p.width, p.height);

        // --- Show the result ---
        p.background(0);
        p.image(pgFade, -p.width / 2, -p.height / 2);

        // --- Swap buffers (ping-pong) ---
        pgMain.background(0);
        pgMain.image(pgFade, -p.width / 2, -p.height / 2);
    }

    function drawChaosWalk() {
        pgMain.stroke(255);
        pgMain.stroke(hueOffset, 100, 100, 100);
        hueOffset = (hueOffset + 1) % 360;
        pgMain.strokeWeight(0.25);
        switch (page) {
            case 0:
                chaosWalkDefault(NUM_ITERATIONS);
                break;
            case 1:
                chaosWalkUnique(NUM_ITERATIONS);
                break;
            case 2:
                chaosWalkDoubleUnique(NUM_ITERATIONS);
                break
            case 3:
                chaosWalkNotOpposite(NUM_ITERATIONS);
                break;
            default:
                console.error("Unknown page");
        }
    }

    function chaosWalkDefault(iterations = 10) {
        for (let i = 0; i < iterations; i++) {
            // pick a random pivot
            const randIdx = Math.floor(pivots.length * Math.random());
            const randPivot = pivots[randIdx];

            // create a new point between the current point and the random pivot
            chaosPoint = lerpFn(chaosPoint, randPivot, STEP_RATIO);
            pgMain.point(chaosPoint);
        }
    }

    function chaosWalkUnique(iterations = 10) {
        for (let i = 0; i < iterations; i++) {
            // pick a random pivot, but not the same as the last one
            let randIdx = Math.floor(pivots.length * Math.random());
            if (randIdx !== lastIdx) {
                const randPivot = pivots[randIdx];
                // create a new point between the current point and the random pivot
                chaosPoint = lerpFn(chaosPoint, randPivot, STEP_RATIO);
                pgMain.point(chaosPoint);
            }
            lastIdx = randIdx;
        }
    }

    function chaosWalkDoubleUnique(iterations = 10) {
        for (let i = 0; i < iterations; i++) {
            // pick a random pivot, but not the same as the last one
            let randIdx = Math.floor(pivots.length * Math.random());
            if (randIdx !== lastIdx && randIdx !== secondLastIdx) {
                const randPivot = pivots[randIdx];
                // create a new point between the current point and the random pivot
                chaosPoint = lerpFn(chaosPoint, randPivot, STEP_RATIO);
                pgMain.point(chaosPoint);
            }
            secondLastIdx = lastIdx;
            lastIdx = randIdx;
        }
    }

    function chaosWalkNotOpposite(iterations = 10) {
        for (let i = 0; i < iterations; i++) {
            // pick a random pivot, but it has to be 2 places away from the last one
            let randIdx = Math.floor(pivots.length * Math.random());
            const oppositeIdx = (lastIdx + NUM_PIVOTS / 2) % NUM_PIVOTS;
            if (randIdx !== oppositeIdx) {
                const randPivot = pivots[randIdx];
                // create a new point between the current point and the random pivot
                chaosPoint = lerpFn(chaosPoint, randPivot, STEP_RATIO);
                pgMain.point(chaosPoint);
            }
            lastIdx = randIdx;
        }
    }

    p.keyPressed = (event: KeyboardEvent) => {
        console.log("keyPressed:", event);
        if (event.code === "Space") {
            initScene(); // reset the scene on space key press
        }
        if (event.code === "KeyZ") {
            accumulate = !accumulate;
        }
        if (event.code === "ArrowDown") {
            if (page > 0) {
                page--;
                pgMain.background(0);
            }
        } else if (event.code === "ArrowUp") {
            if (page < NUM_PAGES - 1) {
                page++;
                pgMain.background(0);
            }
        }
        if (event.code === "Comma") {
            if (lerpId > 0) {
                lerpId--;
                lerpFn = lerpFnMap.get(lerpId) as LerpFunction;
                pgMain.background(0);
            }
        } else if (event.code === "Period") {
            if (lerpId < lerpFnMap.size - 1) {
                lerpId++;
                lerpFn = lerpFnMap.get(lerpId) as LerpFunction;
                pgMain.background(0);
            }
        }
    }

    function basicLerp(a: p5.Vector, b: p5.Vector, t: number) {
        return p5.Vector.lerp(a, b, t);
    }

    function bezierLerp(a: p5.Vector, b: p5.Vector, t: number) {
        const mid = p5.Vector.add(a, b).mult(0.5);
        const offset = p.createVector(-(b.y - a.y), b.x - a.x).normalize().mult(50);
        const control = p5.Vector.add(mid, offset);

        const ab = p5.Vector.lerp(a, control, t);
        const bc = p5.Vector.lerp(control, b, t);
        return p5.Vector.lerp(ab, bc, t);
    }

    function noisyBezierLerp(a: p5.Vector, b: p5.Vector, t: number, noiseScale = 0.01) {
        const mid = p5.Vector.add(a, b).mult(0.5);
        const dir = p5.Vector.sub(b, a);
        const normalDir = p.createVector(-dir.y, dir.x).normalize();

        const n = p.noise(mid.x * noiseScale, mid.y * noiseScale);
        const control = p5.Vector.add(mid, normalDir.mult(n * 100 - 50));

        const ab = p5.Vector.lerp(a, control, t);
        const bc = p5.Vector.lerp(control, b, t);
        return p5.Vector.lerp(ab, bc, t);
    }

    function cubicBezierLerp(a: p5.Vector, b: p5.Vector, t: number) {
        const normalDir = p.createVector(b.y - a.y, -(b.x - a.x)).normalize().mult(50);
        const c1 = p5.Vector.add(chaosPoint, normalDir);
        const c2 = p5.Vector.add(b, normalDir.mult(-1));
        const ab = p5.Vector.lerp(a, c1, t);
        const bc = p5.Vector.lerp(c1, c2, t);
        const cd = p5.Vector.lerp(c2, b, t);
        const abc = p5.Vector.lerp(ab, bc, t);
        const bcd = p5.Vector.lerp(bc, cd, t);
        return p5.Vector.lerp(abc, bcd, t);
    }

    function arcLerp(a: p5.Vector, b: p5.Vector, t: number, curveAmount = 0.5) {
        // Midpoint
        const mid = p5.Vector.add(a, b).mult(0.5);

        // Perpendicular vector
        const dir = p5.Vector.sub(b, a);
        const perp = p.createVector(-dir.y, dir.x).normalize();

        // Curve point: push midpoint outward along perpendicular
        const control = p5.Vector.add(mid, perp.mult(curveAmount * dir.mag()));

        // Quadratic Bézier interpolation
        const ab = p5.Vector.lerp(a, control, t);
        const bc = p5.Vector.lerp(control, b, t);
        return p5.Vector.lerp(ab, bc, t);
    }

    function catmullRomLerp(p0: p5.Vector, p1: p5.Vector, t: number): p5.Vector {
        const p2 = p.random(pivots);
        const p3 = p.random(pivots);

        const t2 = t * t;
        const t3 = t2 * t;

        const x = 0.5 * (
            2 * p1.x +
            (-p0.x + p2.x) * t +
            (2*p0.x - 5*p1.x + 4*p2.x - p3.x) * t2 +
            (-p0.x + 3*p1.x - 3*p2.x + p3.x) * t3
        );

        const y = 0.5 * (
            2 * p1.y +
            (-p0.y + p2.y) * t +
            (2*p0.y - 5*p1.y + 4*p2.y - p3.y) * t2 +
            (-p0.y + 3*p1.y - 3*p2.y + p3.y) * t3
        );

        return p.createVector(x, y);
    }

    function perlinLerp(a: p5.Vector, b: p5.Vector, t: number, noiseStrength = 400, noiseScale = 2, z = 0) {
        // Linear interpolation base point
        const base = p5.Vector.lerp(a, b, t);

        // Direction from a to b
        const dir = p5.Vector.sub(b, a);
        const normalDir = p.createVector(-dir.y, dir.x).normalize();

        // Perlin noise offset
        const noiseVal = p.noise(t * noiseScale, z);
        const offset = normalDir.mult((noiseVal - 0.5) * 2 * noiseStrength);

        return base.add(offset);
    }

    // todo: you can modulate the phase to create a moving wave effect
    function sinusoidalLerp(a: p5.Vector, b: p5.Vector, t: number, amplitude = 200, frequency = 1, phase = Math.PI/4) {
        // Base linear interpolation
        const base = p5.Vector.lerp(a, b, t);

        // Direction and perpendicular
        const dir = p5.Vector.sub(b, a);
        const normalDir = p.createVector(-dir.y, dir.x).normalize();

        // Sine-based offset
        // const wave = Math.sin(2 * Math.PI * frequency * t + (p.millis() / 1000) / 20);
        const wave = Math.sin(2 * Math.PI * frequency * t + phase);
        const offset = normalDir.mult(wave * amplitude);

        return base.add(offset);
    }

    function magnetLerp(from: p5.Vector, to: p5.Vector, t: number, strength = 200.0) {
        const base = p5.Vector.lerp(from, to, t);
        const magnet = p.createVector(0, 0); // center of the canvas as the magnet
        const toMagnet = p5.Vector.sub(magnet, base);
        const radius = toMagnet.mag();
        const towardMagnet = toMagnet.mult(strength / radius);
        return base.add(towardMagnet);
    }

    function perlinMagnetLerp(a: p5.Vector, b: p5.Vector, t: number) {
        const pt1 = perlinLerp(a, b, t);
        return magnetLerp(a, pt1, t);
    }

    function experimentLerp(a: p5.Vector, b: p5.Vector, t: number) {
        const pt1 = cubicBezierLerp(a, b, t);
        const pt2 = perlinLerp(a, pt1, t);
        return magnetLerp(a, pt2, t);
    }

    type P5Slider = {
        input(cb: () => void): void;
        changed(cb: () => void): void;
        value(): number;
    } & p5.Element;

    type LerpFunction = (a: p5.Vector, b: p5.Vector, t: number, ...args: any[]) => p5.Vector;
});
