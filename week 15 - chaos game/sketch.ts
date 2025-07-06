import p5 from 'p5';

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

    let accumulate = true; // whether to accumulate points or not
    let page = 0; // current page for different chaos walk methods
    let lerpFn: LerpFunction = lerp; // default lerp function
    let lerpId = 0; // id of the lerp function
    let lerpFnMap: Map<number, LerpFunction> = new Map([
        [0, lerp],
        [1, arcLerp],
        [2, bezierLerp],
        [3, cubicBezierLerp],
        [4, noisyBezierLerp],
    ]);

    p.setup = () => {
        p.createCanvas(1080, 720, p.WEBGL);
        p.colorMode(p.HSB, 360, 100, 100, 100)

        initScene();
        initUI();
    }

    function initUI() {
        let line1 = p.createDiv();
        p.createSpan("Step Ratio:").parent(line1);
        const stepRatioSlider: P5Slider = p.createSlider(0, 2, STEP_RATIO, 0.005).parent(line1) as P5Slider;
        stepRatioSlider.input(() => {
            STEP_RATIO = stepRatioSlider.value();
            p.background(0);
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

        lerpFn = lerpFnMap.get(lerpId) as LerpFunction;

        chaosPoint = p.createVector(p.random(p.width), p.random(p.height));
        p.background(0);
    }

    p.draw = () => {
        drawPivots();

        p.stroke(hueOffset, 100, 100, 100);
        hueOffset = (hueOffset + 1) % 360;
        p.strokeWeight(0.25);
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

    function drawPivots() {
        p.stroke(255);
        p.strokeWeight(8);
        for (const pivot of pivots) {
            p.point(pivot.x, pivot.y);
        }
    }

    function chaosWalkDefault(iterations = 10) {
        for (let i = 0; i < iterations; i++) {
            // pick a random pivot
            const randIdx = Math.floor(pivots.length * Math.random());
            const randPivot = pivots[randIdx];

            // create a new point between the current point and the random pivot
            chaosPoint = lerpFn(chaosPoint, randPivot, STEP_RATIO);
            p.point(chaosPoint);
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
                p.point(chaosPoint);
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
                p.point(chaosPoint);
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
                p.point(chaosPoint);
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
                p.background(0);
            }
        } else if (event.code === "ArrowUp") {
            if (page < NUM_PAGES - 1) {
                page++;
                p.background(0);
            }
        }
        if (event.code === "Comma") {
            if (lerpId > 0) {
                lerpId--;
                lerpFn = lerpFnMap.get(lerpId) as LerpFunction;
                p.background(0);
            }
        } else if (event.code === "Period") {
            if (lerpId < lerpFnMap.size - 1) {
                lerpId++;
                lerpFn = lerpFnMap.get(lerpId) as LerpFunction;
                p.background(0);
            }
        }
    }

    function lerp(a: p5.Vector, b: p5.Vector, t: number) {
        return p5.Vector.lerp(a, b, t);
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
        const normal = p.createVector(-dir.y, dir.x).normalize();

        const n = p.noise(mid.x * noiseScale, mid.y * noiseScale);
        const control = p5.Vector.add(mid, normal.mult(n * 100 - 50));

        const ab = p5.Vector.lerp(a, control, t);
        const bc = p5.Vector.lerp(control, b, t);
        return p5.Vector.lerp(ab, bc, t);
    }

    function cubicBezierLerp(a: p5.Vector, b: p5.Vector, t: number) {
        const normal = p.createVector(b.y - a.y, -(b.x - a.x)).normalize().mult(50);
        const c1 = p5.Vector.add(chaosPoint, normal);
        const c2 = p5.Vector.add(b, normal.mult(-1));
        const ab = p5.Vector.lerp(a, c1, t);
        const bc = p5.Vector.lerp(c1, c2, t);
        const cd = p5.Vector.lerp(c2, b, t);
        const abc = p5.Vector.lerp(ab, bc, t);
        const bcd = p5.Vector.lerp(bc, cd, t);
        return p5.Vector.lerp(abc, bcd, t);
    }

    type P5Slider = {
        input(cb: () => void): void;
        changed(cb: () => void): void;
        value(): number;
    } & p5.Element;

    type LerpFunction = (a: p5.Vector, b: p5.Vector, t: number) => p5.Vector;
});
