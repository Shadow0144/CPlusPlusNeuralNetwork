#include "NeuralNetwork.h"

#include <iostream>
#include <math.h>
#include <cmath>

NeuralNetwork::NeuralNetwork(int layerCount, int* layerShapes, ActivationFunction* layerFunctions)
{
	this->verbosity = 1;
	this->drawRate = 100;
	this->drawingEnabled = true;
	this->layerCount = layerCount;
	this->layerShapes = layerShapes;
	layers = new vector<Neuron*>[layerCount];
	vector<Neuron*>* parents = NULL;
	for (int i = 0; i < layerCount; i++)
	{
		layers[i] = vector<Neuron*>();
		for (int j = 0; j < layerShapes[i]; j++)
		{
			layers[i].push_back(new Neuron(layerFunctions[i], parents));
		}
		parents = &layers[i];
	}

	this->errorFunction = NULL;
	this->maxIterations = 10000;
	this->minError = 0.001;
	this->errorConvergenceThreshold = 0.00000001;
	this->weightConvergenceThreshold = 0.001;
}

NeuralNetwork::~NeuralNetwork()
{
	//delete layers;
}

MatrixXd NeuralNetwork::feedForward(MatrixXd inputs)
{
	// Fix: Assumes a non-zero network
	MatrixXd result = MatrixXd(inputs.rows(), layerShapes[layerCount-1]); // The final result encapsulating all examples
	if (layerCount > 0 && inputs.cols() == 1 /*input.cols == layerShapes[0]*/)
	{
		for (int i = 0; i < inputs.rows(); i++) // Loop through the examples
		{
			MatrixXd example = inputs.row(i); // A single example
			for (int j = 0; j < layerCount; j++) // Loop through the layers
			{
				MatrixXd row_result = MatrixXd(1, layerShapes[j]); // The result of a single layer of calculation
				for (int k = 0; k < layerShapes[j]; k++) // Loop through the neurons
				{
					row_result(k) = layers[j].at(k)->feedForward(example)(0);
				}
				example = row_result; // Feed the row_result into the next row
			}
			result.row(i) = example;
		}
	}
	else { }

	return result;
}

bool NeuralNetwork::backPropagate(MatrixXd inputs, MatrixXd targets)
{
	bool converged = true;
	if (inputs.rows() == targets.rows())
	{
		for (int i = 0; i < targets.rows(); i++)
		{
			MatrixXd y = feedForward(inputs.row(i));
			MatrixXd errors = (targets.row(i) - y).colwise().sum();
			for (int j = (layerCount - 1); j > 0; j--) // Skip the top row
			{
				MatrixXd newErrors = MatrixXd(layerShapes[j], layerShapes[j - 1]); // One row per neuron above
				for (int k = 0; k < layerShapes[j]; k++)
				{
					MatrixXd newError = layers[j].at(k)->backPropagate(errors.col(k));
					newErrors.row(k) = newError.row(0);
				}
				errors = newErrors;
			}
			for (int k = 0; k < layerShapes[0]; k++) // Top row
			{
				layers[0].at(k)->backPropagate(errors.col(k));
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
	const int SIZE = 600;
	char window_name[] = "Neural Network";
	//MatrixXd img = Mat::Mat(SIZE, SIZE, CV_8UC3, Scalar(225, 225, 225));
	
	/*DrawingCanvas canvas;
	canvas.canvas = img;
	canvas.offset = Point(0, 0);
	canvas.scale = 1.0f;*/

	MatrixXd predicted = feedForward(inputs);
	//draw(canvas, predicted, targets);

	double error = getError(predicted, targets);
	if (verbosity >= 1)
	{
		cout << "Initial: " << endl;
		if (verbosity >= 2)
		{
			for (int i = 0; i < inputs.rows(); i++)
			{
				cout << "Feedforward Untrained: X: " << inputs(i) << " Y': " << targets(i) << " Y: " << predicted(i) << endl;
			}
		}
		else {}
		cout << "Error: " << error << endl;
	}
	else {}

	/*draw(canvas, inputs, targets);
	imshow(window_name, img);
	moveWindow(window_name, 400, 180);
	waitKey(100);*/

	int t = 0;
	bool converged = false;
	double lastError = error;
	double deltaError = error;
	while (error > minError && t < maxIterations && !converged && deltaError > errorConvergenceThreshold)
	{
		converged = backPropagate(inputs, targets);
		predicted = feedForward(inputs);
		error = getError(predicted, targets);
		deltaError = abs(lastError - error);
		lastError = error;

		if (t % drawRate == 0)
		{
			/*draw(canvas, inputs, targets);
			imshow(window_name, img);
			waitKey(1); // Wait enough for the window to draw*/
			if (verbosity >= 1)
			{
				cout << endl << "Iterations: " << (t + 1) << endl;
				if (verbosity >= 2)
				{
					for (int i = 0; i < inputs.rows(); i++)
					{
						cout << "Feedforward Training: X: " << inputs(i)
							<< " Y': " << targets(i)
							<< " Y: " << predicted(i) << endl;
					}
				}
				else {}
				cout << "Error: " << error << endl;
			}
			else {}
		}

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
		else {}
	}
	else {}

	if (verbosity >= 1)
	{
		cout << endl << "Trained: " << endl;
		if (verbosity >= 2)
		{
			for (int i = 0; i < inputs.rows(); i++)
			{
				cout << "Feedforward Trained: X: " << inputs(i) << " Y': " << targets(i) << " Y: " << predicted(i) << endl;
			}
		}
		else {}
		cout << "Iterations: " << t << endl;
		cout << "Error: " << getError(predicted, targets) << endl;
	}
	else {}

	/*draw(canvas, inputs, targets);
	imshow(window_name, img);
	cout << endl << "Press any key to exit" << endl;
	waitKey(0); // Wait for a keystroke in the window*/
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

int NeuralNetwork::getDrawRate()
{
	return drawRate;
}

void NeuralNetwork::setDrawRate(int drawRate)
{
	this->drawRate = drawRate;
}

bool NeuralNetwork::getDrawingEnabled()
{
	return drawingEnabled;
}

void NeuralNetwork::setDrawingEnabled(bool drawingEnabled)
{
	this->drawingEnabled = drawingEnabled;
}

bool windowMade = false;
void NeuralNetwork::draw(MatrixXd target_xs, MatrixXd target_ys)
{
	/*// Clear the canvas
	canvas.canvas.setTo(Scalar(225, 225, 225));

	// Calculate the drawing space parameters
	const int HALF_WIDTH = canvas.canvas.cols / 2;
	const int HALF_HEIGHT = canvas.canvas.rows / 2;
	const int Y_SHIFT = 120;
	const int X_SHIFT = 120;
	int y = 0;
	int x = 0;

	int previous_y = 0;
	int* previous_xs = new int[0];
	int* next_xs;
	int previous_count = 0;
	
	// Find the vertical start point
	y = HALF_HEIGHT - (layerCount * Y_SHIFT / 2) - (Y_SHIFT / 2);
	for (int i = 0; i < layerCount; i++)
	{
		// Find the horizontal start point for this layer
		x = HALF_WIDTH - (layerShapes[i] * X_SHIFT / 2) - (X_SHIFT / 2);
		next_xs = new int[layerShapes[i]];
		for (int j = 0; j < layerShapes[i]; j++)
		{
			// Set the offset and draw the neuron
			canvas.offset = Point(x, y);
			layers[i][j]->draw(canvas, (i == (layerCount-1)));
			next_xs[j] = x;
			x += X_SHIFT;
		}

		previous_count = layerShapes[i];

		delete[] previous_xs;
		previous_xs = next_xs;

		previous_y = y;
		y += Y_SHIFT;
	}

	// Draw the function approximated by the network
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