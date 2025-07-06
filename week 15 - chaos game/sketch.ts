import p5 from 'p5';

new p5((p: p5) => {
    let NUM_PIVOTS = 5; // number of pivots
    let STEP_RATIO = 0.5; // how far to step towards the pivot
    const NUM_ITERATIONS = 5000; // number of iterations for the chaos walk
    const NUM_PAGES = 2; // number of different chaos walk methods

    let pivots: p5.Vector[] = [];
    let chaosPoint: p5.Vector;
    let hueOffset = 0;

    let accumulate = true; // whether to accumulate points or not
    let page = 0; // current page for different chaos walk methods

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
        if (event.code === "Z") {
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
    }

    type P5Slider = {
        input(cb: () => void): void;
        changed(cb: () => void): void;
        value(): number;
    } & p5.Element;
});
