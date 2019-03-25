require('@tensorflow/tfjs-node');
const plot = require('node-remote-plot');
const _ = require('lodash');
const loadCSV = require('../data/load-csv');
const LogisticRegression = require('./logistic-regression');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    converters: {
        mpg: (value) => {
            const mpg = parseFloat(value);

            if (mpg < 15) {
                return [1, 0, 0];
            } else if (mpg < 30) {
                return [0, 1, 0];
            } else {
                return [0, 0, 1];
            }
        }
    },
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg']
});

const regression = new LogisticRegression(features, _.flatMap(labels), {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 10
});

regression.train();

const test = regression.test(testFeatures, _.flatMap(testLabels));
console.log(test);

// plot({
//     x: regression.costHistory.reverse()
// });