"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem


import shared.filt.TestTrainSplitFilter as TestTrainSplitFilter
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

INPUT_FILE = os.path.join("undersampled_fraud.csv")

INPUT_LAYER = 30
HIDDEN_LAYER = 32
OUTPUT_LAYER = 1
# TRAINING_ITERATIONS = 2100


def initialize_instances():
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []

    # Read in the abalone.txt CSV file
    with open(INPUT_FILE, "r") as fraud:
        reader = csv.reader(fraud, delimiter='\t')
        firstRow = True
        for row in reader:
            if firstRow: 
                firstRow = False
                continue
            
            instance = Instance([float(value) for value in row[1:-1]])
            instance.setLabel(Instance(float(row[-1])))
            instances.append(instance)
            # print(instance)

    return instances


def train(oa, network, oaName, train_instances, test_instances, measure, training_iterations):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print "\nError results for %s\n---------------------------" % (oaName,)
    # csvWriter = csv.writer(open(oaName + '.csv', 'w'), delimiter='\t')
    # csv.writerow(['train_error', 'test_error'])
    for iteration in xrange(training_iterations):
        oa.train()

        train_error = 0.00
        for instance in train_instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            train_error += measure.value(output, example)

        test_error = 0.00
        for instance in test_instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            test_error += measure.value(output, example)

        
        print "%0.03f %0.03f" % (train_error, test_error)


def main():
    """Run algorithms on the abalone dataset."""
    instances = initialize_instances()

    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(instances)

    split_filter = TestTrainSplitFilter(70)
    split_filter.filter(data_set)
    train_set = split_filter.getTrainingSet()
    test_set = split_filter.getTestingSet()

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "SA 2", "GA"]
    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(train_set, classification_network, measure))


    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
    oa.append(SimulatedAnnealing(1E11, .55, nnop[2]))
    oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[3]))

    iters = [3200, 3200, 600]

    for i, name in enumerate(oa_names):
        start = time.time()
        correct = 0
        incorrect = 0

        train(oa[i], networks[i], oa_names[i], train_set, test_set, measure, iters[i])
        end = time.time()
        training_time = end - start

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        start = time.time()
        for instance in test_set:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            predicted = instance.getLabel().getContinuous()
            actual = networks[i].getOutputValues().get(0)

            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1

        end = time.time()
        testing_time = end - start

        results += "\nResults for %s: \nCorrectly classified %d instances." % (name, correct)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        results += "\nTraining time: %0.03f seconds" % (training_time,)
        results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

    print results


if __name__ == "__main__":
    main()

