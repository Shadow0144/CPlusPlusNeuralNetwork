#include "Neuron.h"
#include "DotProductFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

Neuron::Neuron(ActivationFunction function)
{
	functionType = function;
	switch (functionType)
	{
	case ActivationFunction::WeightedDotProduct:
			activationFunction = new DotProductFunction();
			break;
		default:
			activationFunction = new DotProductFunction();
			break;
	}
}

Neuron::~Neuron()
{
	delete activationFunction;
}

float Neuron::feedForward(Mat input)
{
	return 0;
}

float Neuron::backPropagate(float error)
{
	return 0;
}

void Neuron::draw(DrawingCanvas canvas, int* previous_xs, int previous_count, int previous_y, bool output)
{
	const int p = 10;
	const int radius = 20;
	const Scalar black(0, 0, 0);
	const Scalar gray(100, 100, 100);
	const Scalar light_gray(200, 200, 200);
	const int line_length = 15;

	// Draw the neuron
	Point pt = Point(p + radius, p + radius) + canvas.offset;
	circle(canvas.canvas, pt, radius, light_gray, -1, LINE_8);
	circle(canvas.canvas, pt, radius, black, 1, LINE_8);

	// Draw the activation function
	DrawingCanvas innerCanvas;
	innerCanvas.canvas = canvas.canvas;
	innerCanvas.offset = pt;
	innerCanvas.scale = canvas.scale;
	activationFunction->draw(innerCanvas);

	// Draw the weights

	// Draw the links
	pt.y -= radius;
	if (previous_count == 0)
	{
		// Draw the input lines
		Point previous(pt.x, pt.y - line_length);
		line(canvas.canvas, previous, pt, gray, 1, LINE_8);
	}
	else
	{
		// Draw the links to the previous neurons
		for (int i = 0; i < previous_count; i++)
		{
			Point previousNeuron(previous_xs[i] + radius + p, previous_y + (2 * radius) + p);
			line(canvas.canvas, previousNeuron, pt, black, 1, LINE_8);
		}
	}
	if (output)
	{
		// Draw the output lines
		pt.y += (2 * radius);
		Point next(pt.x, pt.y + line_length);
		line(canvas.canvas, next, pt, gray, 1, LINE_8);
	}
	else { }
}