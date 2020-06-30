#define _USE_MATH_DEFINES

#include "Neuron.h"
#include "IdentityFunction.h"
#include "DotProductFunction.h"
#include "ReLUFunction.h"
#include "SigmoidFunction.h"
#include "TanhFunction.h"

#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

Neuron::Neuron(ActivationFunction function, vector<Neuron*>* parents)
{
	this->parents = parents;
	if (parents != NULL)
	{
		this->parentCount = ((int)(parents->size()));
	}
	else
	{
		this->parentCount = 0;
	}
	this->children = new vector<Neuron*>();
	childCount = 0;
	for (int i = 0; i < parentCount; i++)
	{
		this->parents->at(i)->addChild(this);
	}
	functionType = function;
	inputCount = (parentCount == 0) ? 1 : parentCount;
	switch (functionType)
	{
		case ActivationFunction::Identity:
			activationFunction = new IdentityFunction(inputCount);
			break;
		case ActivationFunction::WeightedDotProduct:
			inputCount++; // Bias
			activationFunction = new DotProductFunction(inputCount);
			break;
		case ActivationFunction::ReLU:
			inputCount++; // Bias
			activationFunction = new ReLUFunction(inputCount);
			break;
		case ActivationFunction::Sigmoid:
			inputCount++; // Bias
			activationFunction = new SigmoidFunction(inputCount);
			break;
		case ActivationFunction::Tanh:
			inputCount++; // Bias
			activationFunction = new TanhFunction(inputCount);
			break;
		default:
			activationFunction = new IdentityFunction(inputCount);
			break;
	}
}

Neuron::~Neuron()
{
	delete activationFunction;
	delete children;
}

void Neuron::addChild(Neuron* child)
{
	children->push_back(child);
	childCount++;
}

Mat Neuron::feedForward(Mat input)
{
	lastInput = Mat(input);
	if (activationFunction->hasBias())
	{
		hconcat(lastInput, Mat::ones(1, 1, CV_32FC1), lastInput); // Add bias
	}
	else { }
	result = activationFunction->feedForward(lastInput);
	return result;
}

Mat Neuron::backPropagate(Mat errors)
{
	return activationFunction->backPropagate(lastInput, errors);
}

void Neuron::applyBackPropagate()
{
	activationFunction->applyBackProgate();
}

void Neuron::draw(DrawingCanvas canvas, bool output)
{
	const int p = 10;
	const int radius = 40;
	const Scalar black(0, 0, 0);
	const Scalar gray(100, 100, 100);
	const Scalar light_gray(200, 200, 200);
	const int line_length = 15;
	const int weight_radius = 10;

	// Draw the neuron
	drawingParameters.center = Point(p + radius, p + radius) + canvas.offset;
	circle(canvas.canvas, drawingParameters.center, radius, light_gray, -1, LINE_8);
	circle(canvas.canvas, drawingParameters.center, radius, black, 1, LINE_8);

	// Draw the activation function
	DrawingCanvas innerCanvas;
	innerCanvas.canvas = canvas.canvas;
	innerCanvas.offset = Point(drawingParameters.center);
	innerCanvas.scale = canvas.scale;
	activationFunction->draw(innerCanvas);

	// Draw the weights
	double i_angle = 0;
	double angle = 0;
	if (parentCount <= 1)
	{
		i_angle = (M_PI / 2.0);
		angle = 0;
	}
	else if (parentCount % 2 == 0)
	{
		i_angle = (3.0 * M_PI / 4.0);
		angle = M_PI / ((double)(parentCount)) / 2.0;
	}
	else 
	{
		i_angle = (3.0 * M_PI / 4.0);
		angle = M_PI / (((double)(parentCount)) - 1.0) / 2.0;
	}
	for (int i = 0; i < parentCount; i++)
	{
		int w_x = ((int)(cos(i_angle - (angle * i)) * radius + drawingParameters.center.x));
		int w_y = ((int)(-sin(i_angle - (angle * i)) * radius + drawingParameters.center.y));
		Point weight_pt(w_x, w_y);
		//circle(canvas.canvas, weight_pt, weight_radius, black, 1, LINE_8);
		//putText(canvas.canvas, to_string(i), weight_pt, FONT_HERSHEY_COMPLEX_SMALL, 1.0, black);
	}

	// Draw the links
	Point pt = Point(drawingParameters.center);
	pt.y -= radius;
	if (parentCount == 0)
	{
		// Draw the input lines
		Point previous(pt.x, pt.y - line_length);
		line(canvas.canvas, previous, pt, gray, 1, LINE_8);
	}
	else
	{
		// Draw the links to the previous neurons
		for (int i = 0; i < parentCount; i++)
		{
			int previousX = parents->at(i)->drawingParameters.center.x;
			int previousY = parents->at(i)->drawingParameters.center.y;
			Point previousNeuron(previousX, previousY + radius);
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