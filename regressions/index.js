require('@tensorflow/tfjs-node');
const tfjs = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

const { features, labels, testFeatures, testLabels } = loadCSV('cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10
});

regression.train();

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
});

console.log('R2 is', regression.test(testFeatures, testLabels));

regression.predict([
    [120, 2, 380]
]).print();