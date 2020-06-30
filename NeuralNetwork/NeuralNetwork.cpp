#include "NeuralNetwork.h"

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

bool NeuralNetwork::backPropagate(Mat y, Mat yHat)
{
	return false;
}

void NeuralNetwork::draw(DrawingCanvas canvas)
{
	// Calculate the drawing space parameters
	const int half_width = canvas.canvas.cols / 2;
	const int half_height = canvas.canvas.rows / 2;
	const int y_shift = 120;
	const int x_shift = 120;
	int y = 0;
	int x = 0;

	int previous_y = 0;
	int* previous_xs = new int[0];
	int* next_xs;
	int previous_count = 0;
	
	// Find the vertical start point
	y = half_height - (layerCount * y_shift / 2) - (y_shift / 2);
	for (int i = 0; i < layerCount; i++)
	{
		// Find the horizontal start point for this layer
		x = half_width - (layerShapes[i] * x_shift / 2) - (x_shift / 2);
		next_xs = new int[layerShapes[i]];
		for (int j = 0; j < layerShapes[i]; j++)
		{
			// Set the offset and draw the neuron
			canvas.offset = Point(x, y);
			layers[i][j]->draw(canvas, (i == (layerCount-1)));
			next_xs[j] = x;
			x += x_shift;
		}

		previous_count = layerShapes[i];

		delete[] previous_xs;
		previous_xs = next_xs;

		previous_y = y;
		y += y_shift;
	}
}