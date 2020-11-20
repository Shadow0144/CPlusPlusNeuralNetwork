#include "NeuralNetwork.h"
#include "NetworkVisualizer.h"
#include "InputNeuralLayer.h"
#include "DenseNeuralLayer.h"
#include "SoftmaxNeuralLayer.h"
#include "ConvolutionNeuralLayer.h"
#include "PoolingNeuralLayer.h"
#include "FlattenNeuralLayer.h"

#include "Test.h"

#pragma warning(push, 0)
#include <thread>
#include <iostream>
#include <math.h>
#include <cmath>
#include <xtensor/xview.hpp>
#pragma warning(pop)

NeuralNetwork::NeuralNetwork(bool drawingEnabled)
{
	this->verbosity = 1;
	this->batchSize = 1;
	this->outputRate = 100;
	this->drawingEnabled = drawingEnabled;

	this->errorFunction = NULL;
	this->maxIterations = 10000;
	this->minError = 0.001;
	this->errorConvergenceThreshold = 0.00000001;
	this->weightConvergenceThreshold = 0.001;

	colors = NULL;
	if (drawingEnabled)
	{
		visualizer = new NetworkVisualizer(this);
	}
	else 
	{
		visualizer = NULL;
	}

	this->layerCount = 0;
	this->layers = new vector<NeuralLayer*>();
}

NeuralNetwork::~NeuralNetwork()
{
	delete visualizer;
	delete layers;
}

void NeuralNetwork::addInputLayer(std::vector<size_t> inputShape)
{
	this->inputShape = inputShape; 
	InputNeuralLayer* layer = new InputNeuralLayer(inputShape);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addDenseLayer(DenseActivationFunction layerFunction, size_t numUnits)
{
	DenseNeuralLayer* layer = new DenseNeuralLayer(layerFunction, layers->at(layerCount-1), numUnits);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addSoftmaxLayer(int axis)
{
	SoftmaxNeuralLayer* layer = new SoftmaxNeuralLayer(layers->at(layerCount - 1), axis);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addConvolutionLayer(ConvolutionActivationFunction layerFunction, size_t numKernels, std::vector<size_t> convolutionShape, size_t stride)
{
	ConvolutionNeuralLayer* layer = new ConvolutionNeuralLayer(layerFunction, layers->at(layerCount - 1), numKernels, convolutionShape, stride);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addPoolingLayer(PoolingActivationFunction layerFunction, std::vector<size_t> poolingShape)
{
	PoolingNeuralLayer* layer = new PoolingNeuralLayer(layerFunction, layers->at(layerCount - 1), poolingShape);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addFlattenLayer(int numOutputs)
{
	FlattenNeuralLayer* layer = new FlattenNeuralLayer(layers->at(layerCount - 1), numOutputs);
	layers->push_back(layer);
	layerCount++;
}

xt::xarray<double> NeuralNetwork::feedForward(xt::xarray<double> inputs)
{
	xt::xarray<double> predicted = inputs;

	for (int i = 0; i < layerCount; i++) // Loop through the layers
	{
		predicted = layers->at(i)->feedForward(predicted);
	}

	return predicted;
}

xt::xarray<double> NeuralNetwork::feedForwardTrain(xt::xarray<double> inputs)
{
	xt::xarray<double> predicted = inputs;

	for (int i = 0; i < layerCount; i++) // Loop through the layers
	{
		predicted = layers->at(i)->feedForwardTrain(predicted);
	}

	return predicted;
}

bool NeuralNetwork::backPropagate(xt::xarray<double> inputs, xt::xarray<double> targets)
{
	bool converged = true;

	// Feed forward and calculate the gradient
	xt::xarray<double> predicted = feedForwardTrain(inputs);
	xt::xarray<double> errors = errorFunction->getDerivativeOfError(predicted, targets);

	// Backpropagate through the layers
	for (int l = layerCount - 1; l > 0; l--)
	{
		errors = layers->at(l)->backPropagate(errors);
	}

	// Apply the backpropagation
	for (int l = 0; l < layerCount; l++)
	{
		double deltaSum = layers->at(l)->applyBackPropagate();
		converged = (deltaSum < weightConvergenceThreshold) && converged;
	}

	return converged;
}

void NeuralNetwork::train(xt::xarray<double> inputs, xt::xarray<double> targets)
{
	xt::xarray<double> predicted = feedForward(inputs);
	double error = abs(getError(predicted, targets));
	output(LearningState::untrained, 0, inputs, targets, predicted);

	setupDrawing(inputs, targets);
	updateDrawing(predicted);

	cout << "Beginning training" << endl << endl;

	int t = 0;
	bool converged = false;
	double lastError = error;
	double deltaError = error;
	const int BATCHES = inputs.shape()[0] / batchSize;
	int N = targets.shape()[0];
	while ((error < 0 || error > minError) && t < maxIterations && !converged && deltaError > errorConvergenceThreshold)
	{
		converged = true;
		for (int i = 0; i < BATCHES; i++)
		{
			// Set up the batch
			int batchStart = ((i + 0) * batchSize) % N;
			int batchEnd = ((i + 1) * batchSize) % N;
			if ((batchEnd - batchStart) != batchSize)
			{
				batchEnd = N;
			}
			else { }

			xt::xstrided_slice_vector batchSV({ xt::range(batchStart, batchEnd), xt::ellipsis() });
			xt::xarray<double> examples = xt::strided_view(inputs, batchSV);
			xt::xarray<double> exampleTargets = xt::strided_view(targets, batchSV);
			converged = backPropagate(examples, exampleTargets) && converged;
		}

		predicted = feedForward(inputs);
		error = getError(predicted, targets);
		deltaError = abs(lastError - error);
		lastError = error;

		updateDrawing(predicted);

		if (t % outputRate == 0 && t != 0)
		{
			output(LearningState::training, t, inputs, targets, predicted);
		}
		else { }

		t++;
	}

	if (verbosity >= 1)
	{
		if (deltaError <= errorConvergenceThreshold)
		{
			cout << "Error has converged" << endl << endl;
		}
		else if (converged)
		{
			cout << "Weights have converged" << endl << endl;
		}
		else if (error >= 0 && error <= minError)
		{
			cout << "Minimum loss condition reached" << endl << endl;
		}
		else if (t == maxIterations)
		{
			cout << "Maximum iterations reached" << endl << endl;
		}
		else { }
	}
	else { }

	output(LearningState::trained, t, inputs, targets, predicted);

	cout << "Training complete" << endl << endl;

	updateDrawing(predicted);

	system("pause");
}

void NeuralNetwork::output(LearningState state, int iteration, xt::xarray<double> inputs, xt::xarray<double> targets, xt::xarray<double> predicted)
{
	if (verbosity >= 1)
	{
		string stateString = "";
		switch (state)
		{
			case LearningState::untrained:
				stateString = "Initial";
				break;
			case LearningState::training:
				stateString = "Training";
				break;
			case LearningState::trained:
				stateString = "Trained";
				break;
		}
		cout << stateString << ": " << endl;
		if (verbosity >= 2)
		{
			const int N = inputs.shape()[0];
			for (int i = 0; i < N; i++)
			{
				cout << "Feedforward " << stateString << ": X: " << inputs(i) << " Y': " << targets(i) << " Y: " << predicted(i) << endl;
			}
		}
		else { }
		cout << "Iterations: " << iteration << endl;
		cout << "Error: " << getError(predicted, targets) << endl;
		cout << endl;
	}
	else { }
}

void NeuralNetwork::setTrainingParameters(ErrorFunction* errorFunction, int maxIterations,
	double minError, double errorConvergenceThreshold, double weightConvergenceThreshold)
{
	this->errorFunction = errorFunction;
	this->maxIterations = maxIterations;
	this->minError = minError;
	this->errorConvergenceThreshold = errorConvergenceThreshold;
	this->weightConvergenceThreshold = weightConvergenceThreshold;
}

double NeuralNetwork::getError(xt::xarray<double> predicted, xt::xarray<double> actual)
{
	return errorFunction->getError(predicted, actual);
}

int NeuralNetwork::getVerbosity()
{
	return verbosity;
}

void NeuralNetwork::setVerbosity(int verbosity)
{
	this->verbosity = verbosity;
}

int NeuralNetwork::getBatchSize()
{
	return batchSize;
}

void NeuralNetwork::setBatchSize(int batchSize)
{
	this->batchSize = batchSize;
}

int NeuralNetwork::getOutputRate()
{
	return outputRate;
}

void NeuralNetwork::setOutputRate(int outputRate)
{
	this->outputRate = outputRate;
}

bool NeuralNetwork::getDrawingEnabled()
{
	return drawingEnabled;
}

void NeuralNetwork::setDrawingEnabled(bool drawingEnabled)
{
	this->drawingEnabled = drawingEnabled;
	if (drawingEnabled && visualizer == NULL)
	{
		visualizer = new NetworkVisualizer(this);
	}
	else if (!drawingEnabled && visualizer != NULL)
	{ 
		delete visualizer;
		visualizer = NULL;
	}
	else { }
}

void NeuralNetwork::setClassificationVisualizationParameters(int rows, int cols, ImColor* classColors)
{
	if (drawingEnabled)
	{
		visualizer->addClassificationVisualization(rows, cols, classColors);
	}
	else { }
}

void NeuralNetwork::displayRegressionEstimation()
{
	if (drawingEnabled)
	{
		visualizer->addFunctionVisualization();
	}
	else { }
}

void NeuralNetwork::displayClassificationEstimation()
{
	if (drawingEnabled)
	{
		colors = new ImColor[3]{ ImColor(1.0f, 0.0f, 0.0f, 1.0f), ImColor(0.0f, 1.0f, 0.0f, 1.0f), ImColor(0.0f, 0.0f, 1.0f, 1.0f) };
		visualizer->addClassificationVisualization(50, 3, colors);
	}
	else { }
}

void NeuralNetwork::setupDrawing(xt::xarray<double> inputs, xt::xarray<double> targets)
{
	if (drawingEnabled)
	{
		visualizer->setTargets(inputs, targets);
	}
	else { }
}

void NeuralNetwork::updateDrawing(xt::xarray<double> predicted)
{
	if (drawingEnabled)
	{
		visualizer->setPredicted(predicted);
	}
	else { }
}

void NeuralNetwork::draw(ImDrawList* canvas, ImVec2 origin, double scale, xt::xarray<double> target_xs, xt::xarray<double> target_ys)
{
	// Calculate the drawing space parameters
	const double Y_SHIFT = scale * 120.0;
	const double X_SHIFT = scale * 120.0;
	
	// Draw the network
	// Find the vertical start point
	double x = origin.x;
	double y = origin.y - (layerCount * Y_SHIFT / 2.0) + (Y_SHIFT / 2.0);
	for (int l = 0; l < layerCount; l++)
	{
		// Set the offset and draw the layer
		ImVec2 offset = ImVec2(x, y);
		layers->at(l)->draw(canvas, offset, scale, (l == (layerCount-1)));
		y += Y_SHIFT;
	}
}