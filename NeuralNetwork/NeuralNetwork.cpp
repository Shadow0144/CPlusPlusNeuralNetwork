#include "NeuralNetwork.h"
#include "NetworkVisualizer.h"

#include <iostream>
#include <math.h>
#include <cmath>

NeuralNetwork::NeuralNetwork(int layerCount, int* layerShapes, ActivationFunction* layerFunctions, int inputCount, int outputCount, bool drawingEnabled)
{
	this->verbosity = 1;
	this->outputRate = 100;
	this->drawingEnabled = drawingEnabled;
	this->layerCount = layerCount;
	this->layerShapes = layerShapes;
	layers = new vector<Neuron*>[layerCount];
	vector<Neuron*>* parents = NULL;
	for (int i = 0; i < layerCount; i++)
	{
		layers[i] = vector<Neuron*>();
		for (int j = 0; j < layerShapes[i]; j++)
		{
			// TODO: Temp
			int input = (i == 0) ? inputCount : parents->size();
			int output = (i == (layerCount - 1)) ? outputCount : -1;
			layers[i].push_back(new Neuron(layerFunctions[i], parents, input, output));
		}
		parents = &layers[i];
	}

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
}

NeuralNetwork::~NeuralNetwork()
{
	//delete layers;
	delete visualizer;
}

MatrixXd NeuralNetwork::feedForward(MatrixXd inputs)
{
	// Fix: Assumes a non-zero network
	MatrixXd results = MatrixXd(inputs.rows(), layerShapes[layerCount-1] * (layers[layerCount-1][0])->getNumOutputs()); // The final result encapsulating all examples
	if (layerCount > 0)
	{
		for (int n = 0; n < inputs.rows(); n++) // Loop through the examples
		{
			MatrixXd example = inputs.row(n); // A single example
			for (int i = 0; i < layerCount; i++) // Loop through the layers
			{
				int index = 0;
				int width = (layers[i][0])->getNumOutputs();
				MatrixXd row_result = MatrixXd(1, layerShapes[i] * width); // The result of a single layer of calculation
				for (int j = 0; j < layerShapes[i]; j++) // Loop through the neurons
				{
					row_result.block(0, index, 1, width) = layers[i].at(j)->feedForward(example);
					index += width;
				}
				example = row_result; // Feed the row_result into the next row
			}
			results.row(n) = example;
		}
	}
	else { }

	return results;
}

bool NeuralNetwork::backPropagate(MatrixXd inputs, MatrixXd targets)
{
	bool converged = true;
	if (inputs.rows() == targets.rows())
	{
		for (int i = 0; i < targets.rows(); i++)
		{
			MatrixXd y = feedForward(inputs.row(i));
			MatrixXd errors = errorFunction->getDerivativeOfError(y, targets.row(i));
			for (int j = (layerCount - 1); j > 0; j--) // Skip the top row
			{
				MatrixXd newErrors = MatrixXd(layerShapes[j - 1], layerShapes[j]); // One row per previous neuron, one col per current neuron
				for (int k = 0; k < layerShapes[j]; k++)
				{
					MatrixXd newError = layers[j].at(k)->backPropagate(errors.row(k)); // One row per previous neuron
					newErrors.col(k) = newError.col(0); // Each neuron of the current layer returns a col weighted of errors
				}
				errors = newErrors;
			}
			for (int k = 0; k < layerShapes[0]; k++) // Top row (ignore return)
			{
				layers[0].at(k)->backPropagate(errors.row(k));
			}
			
			for (int j = 0; j < layerCount; j++)
			{
				for (int k = 0; k < layerShapes[j]; k++)
				{
					double deltaW = layers[j].at(k)->applyBackPropagate();
					if (deltaW > weightConvergenceThreshold) // Always positive
					{
						converged = false;
					}
					else { }
				}
			}
		}
	}
	else { }

	return converged;
}

void NeuralNetwork::train(MatrixXd inputs, MatrixXd targets)
{
	MatrixXd predicted = feedForward(inputs);
	double error = abs(getError(predicted, targets));
	Output(LearningState::untrained, 0, inputs, targets, predicted);

	cout << "Beginning training" << endl;

	int t = 0;
	bool converged = false;
	double lastError = error;
	double deltaError = error;
	while ((error < 0 || error > minError) && t < maxIterations && !converged && deltaError > errorConvergenceThreshold)
	{
		converged = backPropagate(inputs, targets);
		predicted = feedForward(inputs);
		error = getError(predicted, targets);
		deltaError = abs(lastError - error);
		lastError = error;

		if (t % outputRate == 0 && t != 0)
		{
			Output(LearningState::training, t, inputs, targets, predicted);
		}
		else { }

		if (drawingEnabled)
		{
			visualizer->draw(&predicted, &targets);
		}
		else { }

		t++;
	}
	predicted = feedForward(inputs);

	if (verbosity >= 1)
	{
		if (deltaError <= errorConvergenceThreshold)
		{
			cout << endl << "Error has converged" << endl;
		}
		else if (converged)
		{
			cout << endl << "Weights have converged" << endl;
		}
		else if (error <= minError)
		{
			cout << endl << "Minimum loss condition reached" << endl;
		}
		else if (t == maxIterations)
		{
			cout << endl << "Maximum iterations reached" << endl;
		}
		else { }
	}
	else { }

	Output(LearningState::trained, t, inputs, targets, predicted);

	cout << "Training complete" << endl;

	// Infinite loop
	if (drawingEnabled)
	{
		while (!visualizer->getWindowClosed())
		{
			visualizer->draw(&predicted, &targets);
		}
	}
	else 
	{ 
		cout << endl;
		system("pause");
	}
}

void NeuralNetwork::Output(LearningState state, int iteration, MatrixXd inputs, MatrixXd targets, MatrixXd predicted)
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
		cout << endl << stateString << ": " << endl;
		if (verbosity >= 2)
		{
			for (int i = 0; i < inputs.rows(); i++)
			{
				cout << "Feedforward " << stateString << ": X: " << inputs(i) << " Y': " << targets(i) << " Y: " << predicted(i) << endl;
			}
		}
		else { }
		cout << "Iterations: " << iteration << endl;
		cout << "Error: " << getError(predicted, targets) << endl;
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

double NeuralNetwork::getError(MatrixXd predicted, MatrixXd actual)
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

void NeuralNetwork::draw(ImDrawList* canvas, ImVec2 origin, double scale, MatrixXd target_xs, MatrixXd target_ys)
{
	// Calculate the drawing space parameters
	const double HALF_WIDTH = scale * 1280 / 2;
	const double HALF_HEIGHT = scale * 720 / 2;
	const double Y_SHIFT = scale * 120;
	const double X_SHIFT = scale * 120;
	double y = 0;
	double x = 0;

	double previous_y = 0;
	double* previous_xs = new double[0];
	double* next_xs;
	double previous_count = 0;
	
	// Find the vertical start point
	y = origin.y + HALF_HEIGHT - (layerCount * Y_SHIFT / 2) - (Y_SHIFT / 2);
	for (int i = 0; i < layerCount; i++)
	{
		// Find the horizontal start point for this layer
		x = origin.x + HALF_WIDTH - (layerShapes[i] * X_SHIFT / 2) - (X_SHIFT / 2);
		next_xs = new double[layerShapes[i]];
		for (int j = 0; j < layerShapes[i]; j++)
		{
			// Set the offset and draw the neuron
			ImVec2 offset = ImVec2(x, y);
			layers[i][j]->draw(canvas, offset, scale, (i == (layerCount-1)));
			next_xs[j] = x;
			x += X_SHIFT;
		}

		previous_count = layerShapes[i];

		delete[] previous_xs;
		previous_xs = next_xs;

		previous_y = y;
		y += Y_SHIFT;
	}

	/*// Draw the function approximated by the network
	const int BUFFER = 5;
	const int GRID_SIZE = 150;
	const int LEFT = canvas.canvas.cols - BUFFER - GRID_SIZE;
	const int TOP = canvas.canvas.rows - BUFFER - GRID_SIZE;
	const int CENTER_X = LEFT + (GRID_SIZE / 2);
	const int CENTER_Y = TOP + (GRID_SIZE / 2);
	const Scalar WHITE(255, 255, 255);
	
	rectangle(canvas.canvas, Rect(LEFT, TOP, GRID_SIZE, GRID_SIZE), WHITE, -1, LINE_8);

	const int DOT_LENGTH = 4;
	const int DARK_GRAY = 50;
	Point zero_x_left(LEFT, CENTER_Y);
	Point zero_x_right(LEFT + GRID_SIZE, CENTER_Y);
	LineIterator itX(canvas.canvas, zero_x_left, zero_x_right, LINE_8);
	Point zero_y_base(CENTER_X, TOP);
	Point zero_y_top(CENTER_X, TOP + GRID_SIZE);
	LineIterator itY(canvas.canvas, zero_y_base, zero_y_top, LINE_8);
	for (int i = 0; i < itX.count; i++, itX++, itY++)
	{
		if (i % DOT_LENGTH != 0)
		{
			(*itX)[0] = DARK_GRAY;
			(*itX)[1] = DARK_GRAY;
			(*itX)[2] = DARK_GRAY;
			(*itY)[0] = DARK_GRAY;
			(*itY)[1] = DARK_GRAY;
			(*itY)[2] = DARK_GRAY;
		}
		else { }
	}

	float w = ((int)((target_xs.at<float>(target_xs.rows - 1) - target_xs.at<float>(0)))) / 2.0f;
	int resizeX = ((int)(GRID_SIZE / w));
	// Fix: Clipping y
	const Scalar BLUE(255, 0, 0);
	const float STEP_SIZE = 0.1f;
	MatrixXd input1 = MatrixXd(1, layerShapes[0], CV_32FC1);
	MatrixXd input2 = MatrixXd(1, layerShapes[0], CV_32FC1);
	for (float i = -1.0f; i < 1.0f; i += STEP_SIZE)
	{
		input1.at<float>(0) = i;
		input2.at<float>(0) = i + STEP_SIZE;
		int y1 = ((int)(resizeX * feedForward(input1).at<float>(0)));
		int y2 = ((int)(resizeX * feedForward(input2).at<float>(0)));
		Point l_start(CENTER_X + ((int)(resizeX * i)), CENTER_Y - y1);
		Point l_end(CENTER_X + ((int)(resizeX * (i + STEP_SIZE))), CENTER_Y - y2);
		line(canvas.canvas, l_start, l_end, BLUE, 1, LINE_8);
	}

	const int TARGET_RADIUS = 1;
	const Scalar TARGET_COLOR(0, 0, 255);
	int num_targets = target_xs.rows;
	for (int i = 0; i < num_targets; i++)
	{
		circle(canvas.canvas,
			Point(CENTER_X + ((int)(resizeX * target_xs.at<float>(i))), CENTER_X - ((int)(resizeX * target_ys.at<float>(i)))),
			TARGET_RADIUS, TARGET_COLOR, 1, LINE_8);
	}*/
}