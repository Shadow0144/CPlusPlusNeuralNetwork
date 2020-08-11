#include "NeuralNetwork.h"
#include "NetworkVisualizer.h"
#include "InputNeuralLayer.h"
#include "DenseNeuralLayer.h"

#pragma warning(push, 0)
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

	if (drawingEnabled)
	{
		visualizer = new NetworkVisualizer(this);
	}
	else 
	{
		visualizer = NULL;
	}

	this->layerCount = 0;
	this->layers = new vector<NeuralLayer*>();\
}

NeuralNetwork::~NeuralNetwork()
{
	delete layers;
	delete visualizer;
}

void NeuralNetwork::addInputLayer(std::vector<size_t> inputShape)
{
	this->inputShape = inputShape; 
	InputNeuralLayer* layer = new InputNeuralLayer(inputShape);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addDenseLayer(ActivationFunction layerFunction, size_t numUnits)
{
	DenseNeuralLayer* layer = new DenseNeuralLayer(layerFunction, layers->at(layerCount-1), numUnits);
	layers->push_back(layer);
	layerCount++;
}

xt::xarray<double> NeuralNetwork::feedForward(xt::xarray<double> inputs)
{
	xt::xarray<double> results = inputs;

	for (int i = 0; i < layerCount; i++) // Loop through the layers
	{
		results = layers->at(i)->feedForward(results);
	}

	return results;
}

bool NeuralNetwork::backPropagate(xt::xarray<double> inputs, xt::xarray<double> targets)
{
	bool converged = true;

	//// Create views for the inputs and outputs
	xt::xarray<double> y = feedForward(inputs);
	xt::xarray<double> errors = errorFunction->getDerivativeOfError(y, targets);

	for (int l = layerCount - 1; l > 0; l--)
	{
		//cout << "Layer: " << l << endl;
		errors = layers->at(l)->backPropagate(errors);
	}

	for (int l = 0; l < layerCount; l++)
	{
		converged = layers->at(l)->applyBackPropagate() && converged;
	}

	return converged;
}

void NeuralNetwork::train(xt::xarray<double> inputs, xt::xarray<double> targets)
{
	auto predicted = feedForward(inputs);
	double error = abs(getError(predicted, targets));
	output(LearningState::untrained, 0, inputs, targets, predicted);

	cout << "Beginning training" << endl << endl;

	int t = 0;
	bool converged = false;
	double lastError = error;
	double deltaError = error;
	while ((error < 0 || error > minError) && t < maxIterations && !converged && deltaError > errorConvergenceThreshold)
	{
		const int BATCHES = inputs.shape()[0] / batchSize;
		for (int i = 0; i < BATCHES; i++)
		{
			xt::xstrided_slice_vector batchSV({ xt::range(i * batchSize, (i + 1) * batchSize), xt::ellipsis() });
			xt::xarray<double> examples = xt::strided_view(inputs, batchSV);
			xt::xarray<double> exampleTargets = xt::strided_view(targets, batchSV);
			converged = backPropagate(examples, exampleTargets);
			predicted = feedForward(examples);
		}
		//error = getError(predicted, targets); // TODO
		//deltaError = abs(lastError - error);
		//lastError = error;

		if (t % outputRate == 0 && t != 0)
		{
			output(LearningState::training, t, inputs, targets, predicted);
		}
		else { }

		if (drawingEnabled)
		{
			visualizer->draw(predicted, targets);
		}
		else { }

		t++;
	}
	predicted = feedForward(inputs);

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

	// Infinite loop
	if (drawingEnabled)
	{
		while (!visualizer->getWindowClosed())
		{
			visualizer->draw(predicted, targets);
		}
	}
	else 
	{ 
		system("pause");
	}
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
	visualizer->addClassificationVisualization(rows, cols, classColors);
}

void NeuralNetwork::draw(ImDrawList* canvas, ImVec2 origin, double scale, xt::xarray<double> target_xs, xt::xarray<double> target_ys)
{
	//// Calculate the drawing space parameters
	//const double HALF_WIDTH = scale * 1280 / 2;
	//const double HALF_HEIGHT = scale * 720 / 2;
	//const double Y_SHIFT = scale * 120;
	//const double X_SHIFT = scale * 120;
	//double y = 0;
	//double x = 0;

	//double previous_y = 0;
	//double* previous_xs = new double[0];
	//double* next_xs;
	//double previous_count = 0;
	//
	//// Find the vertical start point
	//y = origin.y + HALF_HEIGHT - (layerCount * Y_SHIFT / 2) - (Y_SHIFT / 2);
	//for (int i = 0; i < layerCount; i++)
	//{
	//	// Find the horizontal start point for this layer
	//	int shape = layerShapes->at(i);
	//	x = origin.x + HALF_WIDTH - (shape * X_SHIFT / 2) - (X_SHIFT / 2);
	//	next_xs = new double[shape];
	//	for (int j = 0; j < shape; j++)
	//	{
	//		// Set the offset and draw the neuron
	//		ImVec2 offset = ImVec2(x, y);
	//		layers->at(i)[j]->draw(canvas, offset, scale, (i == (layerCount-1)));
	//		next_xs[j] = x;
	//		x += X_SHIFT;
	//	}

	//	previous_count = shape;

	//	delete[] previous_xs;
	//	previous_xs = next_xs;

	//	previous_y = y;
	//	y += Y_SHIFT;
	//}

	///*// Draw the function approximated by the network
	//const int BUFFER = 5;
	//const int GRID_SIZE = 150;
	//const int LEFT = canvas.canvas.cols - BUFFER - GRID_SIZE;
	//const int TOP = canvas.canvas.rows - BUFFER - GRID_SIZE;
	//const int CENTER_X = LEFT + (GRID_SIZE / 2);
	//const int CENTER_Y = TOP + (GRID_SIZE / 2);
	//const Scalar WHITE(255, 255, 255);
	//
	//rectangle(canvas.canvas, Rect(LEFT, TOP, GRID_SIZE, GRID_SIZE), WHITE, -1, LINE_8);

	//const int DOT_LENGTH = 4;
	//const int DARK_GRAY = 50;
	//Point zero_x_left(LEFT, CENTER_Y);
	//Point zero_x_right(LEFT + GRID_SIZE, CENTER_Y);
	//LineIterator itX(canvas.canvas, zero_x_left, zero_x_right, LINE_8);
	//Point zero_y_base(CENTER_X, TOP);
	//Point zero_y_top(CENTER_X, TOP + GRID_SIZE);
	//LineIterator itY(canvas.canvas, zero_y_base, zero_y_top, LINE_8);
	//for (int i = 0; i < itX.count; i++, itX++, itY++)
	//{
	//	if (i % DOT_LENGTH != 0)
	//	{
	//		(*itX)[0] = DARK_GRAY;
	//		(*itX)[1] = DARK_GRAY;
	//		(*itX)[2] = DARK_GRAY;
	//		(*itY)[0] = DARK_GRAY;
	//		(*itY)[1] = DARK_GRAY;
	//		(*itY)[2] = DARK_GRAY;
	//	}
	//	else { }
	//}

	//float w = ((int)((target_xs.at<float>(target_xs.rows - 1) - target_xs.at<float>(0)))) / 2.0f;
	//int resizeX = ((int)(GRID_SIZE / w));
	//// Fix: Clipping y
	//const Scalar BLUE(255, 0, 0);
	//const float STEP_SIZE = 0.1f;
	//MatrixXd input1 = MatrixXd(1, layerShapes[0], CV_32FC1);
	//MatrixXd input2 = MatrixXd(1, layerShapes[0], CV_32FC1);
	//for (float i = -1.0f; i < 1.0f; i += STEP_SIZE)
	//{
	//	input1.at<float>(0) = i;
	//	input2.at<float>(0) = i + STEP_SIZE;
	//	int y1 = ((int)(resizeX * feedForward(input1).at<float>(0)));
	//	int y2 = ((int)(resizeX * feedForward(input2).at<float>(0)));
	//	Point l_start(CENTER_X + ((int)(resizeX * i)), CENTER_Y - y1);
	//	Point l_end(CENTER_X + ((int)(resizeX * (i + STEP_SIZE))), CENTER_Y - y2);
	//	line(canvas.canvas, l_start, l_end, BLUE, 1, LINE_8);
	//}

	//const int TARGET_RADIUS = 1;
	//const Scalar TARGET_COLOR(0, 0, 255);
	//int num_targets = target_xs.rows;
	//for (int i = 0; i < num_targets; i++)
	//{
	//	circle(canvas.canvas,
	//		Point(CENTER_X + ((int)(resizeX * target_xs.at<float>(i))), CENTER_X - ((int)(resizeX * target_ys.at<float>(i)))),
	//		TARGET_RADIUS, TARGET_COLOR, 1, LINE_8);
	//}*/
}