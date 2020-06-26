#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int layerCount, int* layerShapes, ActivationFunction* layerFunctions)
{
	this->layerCount = layerCount;
	this->layerShapes = layerShapes;
	layers = new vector<Neuron*>[layerCount];
	for (int i = 0; i < layerCount; i++)
	{
		layers[i] = vector<Neuron*>();
		for (int j = 0; j < layerShapes[i]; j++)
		{
			layers[i].push_back(new Neuron(layerFunctions[i]));
		}
	}
}

NeuralNetwork::~NeuralNetwork()
{
	//delete layers;
}

Mat NeuralNetwork::feedForward(Mat input)
{
	return Mat::zeros(Size(), 0);
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
	const int y_shift = 50;
	const int x_shift = 50;
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
			layers[i][j]->draw(canvas, previous_xs, previous_count, previous_y, (i == (layerCount-1)));
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