require('@tensorflow/tfjs-node');
const tfjs = require('@tensorflow/tfjs');
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
    iterations: 100
});

regression.train();
console.log('R2 is ', regression.test(testFeatures, testLabels));