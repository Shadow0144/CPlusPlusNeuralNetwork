#define _USE_MATH_DEFINES

#include "Neuron.h"
#include "IdentityFunction.h"
#include "DotProductFunction.h"
#include "ReLUFunction.h"
#include "LeakyReLUFunction.h"
#include "SoftplusFunction.h"
#include "SigmoidFunction.h"
#include "TanhFunction.h"
#include "SoftmaxFunction.h"

#include <math.h>

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
		case ActivationFunction::LeakyReLU:
			inputCount++; // Bias
			activationFunction = new LeakyReLUFunction(inputCount);
			break;
		case ActivationFunction::Softplus:
			inputCount++; // Bias
			activationFunction = new SoftplusFunction(inputCount);
			break;
		case ActivationFunction::Sigmoid:
			inputCount++; // Bias
			activationFunction = new SigmoidFunction(inputCount);
			break;
		case ActivationFunction::Tanh:
			inputCount++; // Bias
			activationFunction = new TanhFunction(inputCount);
			break; 
		case ActivationFunction::Softmax:
			activationFunction = new SoftmaxFunction(inputCount);
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

MatrixXd Neuron::feedForward(MatrixXd input)
{
	lastInput = MatrixXd(input);
	if (activationFunction->hasBias())
	{
		MatrixXd concat(lastInput.rows(), lastInput.cols() + 1);
		concat << lastInput, MatrixXd::Ones(1, 1); // Add bias
		lastInput = concat;
	}
	else { }
	result = activationFunction->feedForward(lastInput);
	return result;
}

MatrixXd Neuron::backPropagate(MatrixXd errors)
{
	return activationFunction->backPropagate(lastInput, errors);
}

double Neuron::applyBackPropagate()
{
	return activationFunction->applyBackProgate();
}

void Neuron::draw(ImDrawList* canvas, bool output)
{
	/*const int P = 10;
	const int RADIUS = 40;
	const Scalar BLACK(0, 0, 0);
	const Scalar GRAY(100, 100, 100);
	const Scalar LIGHT_GRAY(200, 200, 200);
	const Scalar WHITE(255, 255, 255);
	const int LINE_LENGTH = 15;
	const int WEIGHT_RADIUS = 10;
	const int BIAS_OFFSET_X = 40;
	const int BIAS_OFFSET_Y = -52;
	const int BIAS_WIDTH = 20;
	const int BIAS_HEIGHT = 30;
	const int BIAS_TEXT_X = 4;
	const int BIAS_TEXT_Y = 20;

	// Draw the neuron
	drawingParameters.center = Point(P + RADIUS, P + RADIUS) + canvas.offset;
	circle(canvas.canvas, drawingParameters.center, RADIUS, LIGHT_GRAY, -1, LINE_8);
	circle(canvas.canvas, drawingParameters.center, RADIUS, BLACK, 1, LINE_8);

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
		int w_x = ((int)(cos(i_angle - (angle * i)) * RADIUS + drawingParameters.center.x));
		int w_y = ((int)(-sin(i_angle - (angle * i)) * RADIUS + drawingParameters.center.y));
		Point weight_pt(w_x, w_y);
		//circle(canvas.canvas, weight_pt, weight_radius, black, 1, LINE_8);
		//putText(canvas.canvas, to_string(i), weight_pt, FONT_HERSHEY_COMPLEX_SMALL, 1.0, BLACK);
	}

	// Draw the bias
	int bx = 0;
	int by = 0;
	if (activationFunction->hasBias())
	{
		bx = drawingParameters.center.x + BIAS_OFFSET_X;
		by = drawingParameters.center.y + BIAS_OFFSET_Y;
		Point bTL = Point(bx, by);
		Point bBR = Point(bx + BIAS_WIDTH, by + BIAS_HEIGHT);
		Point bias_pt = Point(bx + BIAS_TEXT_X, by + BIAS_TEXT_Y);
		rectangle(canvas.canvas, bTL, bBR, BLACK, 1, LINE_8);
		putText(canvas.canvas, to_string(1), bias_pt, FONT_HERSHEY_COMPLEX_SMALL, 1.0, BLACK);
	}
	else { }

	// Draw the links
	Point pt = Point(drawingParameters.center);
	pt.y -= RADIUS;
	// Draw the links to the previous neurons
	int previousX, previousY;
	for (int i = 0; i < inputCount; i++)
	{
		if (parentCount == 0 && i == 0)
		{
			previousX = pt.x;
			previousY = pt.y - LINE_LENGTH;
		}
		else if (i < (inputCount - 1) || !activationFunction->hasBias())
		{
			previousX = parents->at(i)->drawingParameters.center.x;
			previousY = parents->at(i)->drawingParameters.center.y + RADIUS;
		}
		else
		{
			previousX = bx;
			previousY = by + ((int)(BIAS_HEIGHT / 2.0f));
		}

		Point previousNeuronPt(previousX, previousY);
		Scalar lineColor = Scalar(255, 255, 255);
		int lineWidth = 1;
		float weight = activationFunction->getWeights().getParameters().at<float>(i);
		if (weight >= 0.0f)
		{
			if (weight <= 1.0f)
			{
				lineColor = Scalar(1.0 - ((double)(weight)), 1.0 - ((double)(weight)), 1.0 - ((double)(weight))) * 255;
				lineWidth = ((int)(ceil(weight * lineWidth)));
			}
			else 
			{
				lineColor = Scalar(0.0f, 0.0f, 0.0f) * 255;
				lineWidth = ((int)(ceil(1.0f * lineWidth)));
			}
		}
		else
		{
			if (weight >= -1.0f)
			{
				lineColor = Scalar(0.0f, 0.0f, -weight) * 255;
				lineWidth = ((int)(ceil(-weight * lineWidth)));
			}
			else
			{
				lineColor = Scalar(0.0f, 0.0f, 1.0f) * 255;
				lineWidth = ((int)(ceil(1.0f * lineWidth)));
			}
		}
		if (lineWidth > 0)
		{
			line(canvas.canvas, previousNeuronPt, pt, lineColor, lineWidth, LINE_8);
		}
		else
		{
			line(canvas.canvas, previousNeuronPt, pt, WHITE, 1, LINE_8);
		}
	}

	if (output)
	{
		// Draw the output lines
		pt.y += (2 * RADIUS);
		Point next(pt.x, pt.y + LINE_LENGTH);
		line(canvas.canvas, next, pt, GRAY, 1, LINE_8);
	}
	else { }*/
}