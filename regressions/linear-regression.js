const tf = require('@tensorflow/tfjs');

class LinearRegression {
    constructor(features, labels, options) {
        this.features = features;
        this.labels = labels;
        this.options = {
            learningRate: 0.1,
            iterations: 1000,
            ...options
        };

        this.m = 0;
        this.b = 0;
    }

    gradientDescent() {
        const currentGuessForMPG = this.features.map(row => {
            return this.m * row[0] + this.b;
        });

        const bSlope = currentGuessForMPG.map((guess, i) => {
            return guess - this.labels[i][0];
        }).reduce((total, guess) => {
            return total + guess;
        }, 0) * 2 / this.features.length;

        const mSlope = currentGuessForMPG.map((guess, i) => {
            return -this.features[i][0] * (this.labels[i][0] - guess);
        }).reduce((total, guess) => {
            return total + guess;
        }, 0) * 2 / this.features.length;

        this.m = this.m - mSlope * this.options.learningRate;
        this.b = this.b - bSlope * this.options.learningRate;
    }

    train() {
        for (let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent();
        }
    }

    test() {

    }

    predict() {

    }
}

module.exports = LinearRegression;