require('@tensorflow/tfjs-node');
const plot = require('node-remote-plot');
const loadCSV = require('../data/load-csv');
const LogisticRegression = require('./logistic-regression');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: (value) => {
            return value === 'TRUE' ? 1 : 0;
        }
    },
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions']
});

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 50,
    decisionBoundary: 0.5
});

regression.train();

const test = regression.test(testFeatures, testLabels);
console.log(test);

plot({
    x: regression.costHistory.reverse()
});

// regression.predict([
//     [88, 97, 1.065]
// ]).print();