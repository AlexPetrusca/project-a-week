import p5 from 'p5';

new p5((p: p5) => {
    const WIDTH = 1080;
    const HEIGHT = 720;

    const NUM_PIVOTS = 3; // number of pivots
    const TRACE_LIMIT = 100000; // how many points to keep in the trace
    const STEP_RATIO = 0.5; // how far to step towards the pivot

    let pivots: p5.Vector[] = [];
    let pivotColors: p5.Color[] = [];
    let trace: p5.Vector[] = [];
    let chaosPoint: p5.Vector;

    p.setup = () => {
        p.createCanvas(WIDTH, HEIGHT);
        p.colorMode(p.HSB, 360, 100, 100)

        let _hue = 360 * Math.random();
        for (let i = 0; i < NUM_PIVOTS; i++) {
            pivots.push(p.createVector(p.random(WIDTH), p.random(HEIGHT)));
            pivotColors.push(p.color(_hue, 100, 100));
            _hue = (_hue + 360 / NUM_PIVOTS) % 360; // spread colors evenly around the hue circle
        }

        chaosPoint = p.createVector(p.random(WIDTH), p.random(HEIGHT), 0);
    }

    p.draw = () => {
        p.background(0);

        p.stroke(255);
        p.strokeWeight(8);
        for (const pivot of pivots) {
            p.point(pivot.x, pivot.y);
        }

        p.strokeWeight(1);
        for (const xy of trace) {
            p.stroke(pivotColors[xy.z]);
            p.point(xy.x, xy.y);
        }

        chaosWalk(100);
    }

    function chaosWalk(iterations = 10) {
        for (let i = 0; i < iterations; i++) {
            const randIdx = Math.floor(pivots.length * Math.random());
            const randPivot = pivots[randIdx];
            // create a new point that is halfway between the current point and the random pivot
            const halfX = p.lerp(chaosPoint.x, randPivot.x, STEP_RATIO);
            const halfY = p.lerp(chaosPoint.y, randPivot.y, STEP_RATIO);
            chaosPoint = p.createVector(halfX, halfY, randIdx);
            trace.push(chaosPoint);
        }

        // limit the trace to the last TRACE_LIMIT points
        if (trace.length > TRACE_LIMIT) {
            trace.splice(0, trace.length - TRACE_LIMIT);
        }
    }

    p.keyPressed = (event: KeyboardEvent) => {
        console.log("keyPressed:", event);
        if (event.code === "Space") {
            trace = [];
            pivots = [];
            pivotColors = [];

            let _hue = 360 * Math.random();
            for (let i = 0; i < NUM_PIVOTS; i++) {
                pivots.push(p.createVector(p.random(WIDTH), p.random(HEIGHT)));
                pivotColors.push(p.color(_hue, 100, 100));
                console.log(_hue);
                _hue = (_hue + 360 / NUM_PIVOTS) % 360; // spread colors evenly around the hue circle
            }

            chaosPoint = p.createVector(p.random(WIDTH), p.random(HEIGHT), 0);
            trace.push(chaosPoint);
        }
    }
});
