#include "NeuralNetwork.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

NeuralNetwork::NeuralNetwork(int layerCount, int* layerShapes, ActivationFunction* layerFunctions)
{
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
}

NeuralNetwork::~NeuralNetwork()
{
	//delete layers;
}

Mat NeuralNetwork::feedForward(Mat input)
{
	// Fix: Assumes a non-zero network
	Mat result = Mat(0, layerShapes[layerCount-1], CV_32F); // The final result encapsulating all examples
	if (layerCount > 0 && input.cols == 1 /*input.cols == layerShapes[0]*/)
	{
		for (int i = 0; i < input.rows; i++) // Loop through the examples
		{
			Mat example = input.row(i); // A single example
			for (int j = 0; j < layerCount; j++) // Loop through the layers
			{
				Mat row_result = Mat(1, 0, CV_32F); // The result of a single layer of calculation
				for (int k = 0; k < layerShapes[j]; k++) // Loop through the neurons
				{
					hconcat(row_result, layers[j].at(k)->feedForward(example), row_result);
				}
				example = row_result; // Feed the row_result into the next row
			}
			vconcat(result, example, result);
		}
	}
	else { }

	return result;
}

bool NeuralNetwork::backPropagate(Mat xs, Mat yHats)
{
	if (xs.rows = yHats.rows)
	{
		for (int i = 0; i < yHats.rows; i++)
		{
			Mat y = feedForward(xs.row(i));
			Mat errors = Mat(1, layerShapes[layerCount - 1], CV_32FC1, cv::sum(yHats.row(i) - y));
			for (int j = (layerCount - 1); j > 0; j--) // Skip the top row
			{
				Mat newErrors = Mat(0, layerShapes[j - 1], CV_32FC1);
				for (int k = 0; k < layerShapes[j]; k++)
				{
					Mat newError = layers[j].at(k)->backPropagate(errors.col(k));
					vconcat(newErrors, newError, newErrors);
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
					layers[j].at(k)->applyBackPropagate();
				}
			}
		}
	}
	else { }

	return false;
}

float NeuralNetwork::MSE(Mat ys, Mat yHats)
{
	Mat diff = yHats - ys;
	Mat pow;
	cv::pow(diff, 2, pow);
	float result = 0.5f * ((float)(sum(pow)[0]));
	return result;
}

void NeuralNetwork::draw(DrawingCanvas canvas, Mat target_xs, Mat target_ys)
{
	// Clear the canvas
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

	int resizeX = GRID_SIZE / ((int)((target_xs.at<float>(target_xs.rows - 1) - target_xs.at<float>(0)))) / 2;
	// Fix: Clipping y
	const Scalar BLUE(255, 0, 0);
	const float STEP_SIZE = 0.1f;
	Mat input1 = Mat(1, layerShapes[0], CV_32FC1);
	Mat input2 = Mat(1, layerShapes[0], CV_32FC1);
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
	}
}