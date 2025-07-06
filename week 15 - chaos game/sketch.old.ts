import p5 from 'p5';

new p5((p: p5) => {
    const WIDTH = 1080;
    const HEIGHT = 720;

    const NUM_PIVOTS = 5; // number of pivots
    let STEP_RATIO = 0.5; // how far to step towards the pivot

    let pivots: p5.Vector[] = [];
    let chaosPoint: p5.Vector;
    let hueOffset = 0;

    let accumulate = true; // whether to accumulate points or not

    p.setup = () => {
        p.createCanvas(WIDTH, HEIGHT, p.WEBGL);
        p.colorMode(p.HSB, 360, 100, 100, 100)

        initScene();
        initUI();
    }

    function initUI() {
        const slider: P5Slider = p.createSlider(0, 2, STEP_RATIO, 0.01) as P5Slider;
        slider.input(() => {
            STEP_RATIO = slider.value();
            p.background(0);
        });
    }

    function initScene() {
        pivots = [];

        let phase = 2 * Math.PI * Math.random();
        const radius = (HEIGHT - 100)/2;
        for (let i = 0; i < NUM_PIVOTS; i++) {
            const x = radius * Math.cos(phase);
            const y = radius * Math.sin(phase);
            pivots.push(p.createVector(x, y));
            phase += 2 * Math.PI / NUM_PIVOTS; // spread pivots evenly around a circle
        }

        chaosPoint = p.createVector(p.random(WIDTH), p.random(HEIGHT));
        p.background(0);
    }

    p.draw = () => {
        drawPivots();

        p.stroke(hueOffset, 100, 100, 100);
        hueOffset = (hueOffset + 1) % 360;
        p.strokeWeight(0.25);
        // chaosWalkDefault(5000);
        chaosWalkUnique(5000);
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

            // create a new point that is halfway between the current point and the random pivot
            const halfX = p.lerp(chaosPoint.x, randPivot.x, STEP_RATIO);
            const halfY = p.lerp(chaosPoint.y, randPivot.y, STEP_RATIO);
            chaosPoint = p.createVector(halfX, halfY);
            p.point(chaosPoint);
        }
    }

    let lastIdx: number = 0;
    function chaosWalkUnique(iterations = 10) {
        for (let i = 0; i < iterations; i++) {
            // pick a random pivot, but not the same as the last one
            let randIdx = Math.floor(pivots.length * Math.random());
            if (randIdx !== lastIdx) {
                const randPivot = pivots[randIdx];
                // create a new point that is halfway between the current point and the random pivot
                const halfX = p.lerp(chaosPoint.x, randPivot.x, STEP_RATIO);
                const halfY = p.lerp(chaosPoint.y, randPivot.y, STEP_RATIO);
                chaosPoint = p.createVector(halfX, halfY);
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
        if (event.code === "ArrowUp") {
            accumulate = !accumulate;
        } else if (event.code === "ArrowDown") {
            accumulate = !accumulate;
        }
    }

    type P5Slider = {
        input(cb: () => void): void;
        changed(cb: () => void): void;
        value(): number;
    } & p5.Element;
});
