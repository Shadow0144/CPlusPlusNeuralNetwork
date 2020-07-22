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

void Neuron::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	const double P = 10;
	const double RADIUS = 40;
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);
	const double LINE_LENGTH = 15;
	const double WEIGHT_RADIUS = 10;
	const double BIAS_OFFSET_X = 40;
	const double BIAS_OFFSET_Y = -52;
	const double BIAS_FONT_SIZE = 24;
	const double BIAS_WIDTH = 20;
	const double BIAS_HEIGHT = BIAS_FONT_SIZE;
	const double BIAS_TEXT_X = 4;
	const double BIAS_TEXT_Y = 20;

	position = origin;

	// Draw the neuron
	canvas->AddCircleFilled(ImVec2(origin.x, origin.y), RADIUS * scale, LIGHT_GRAY, 32);
	canvas->AddCircle(ImVec2(origin.x, origin.y), RADIUS * scale, BLACK, 32);

	// Draw the activation function
	activationFunction->draw(canvas, origin, scale);

	/*// Draw the weights
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
		double w_x = (cos(i_angle - (angle * i)) * scale * RADIUS + origin.x);
		double w_y = (-sin(i_angle - (angle * i)) * scale * RADIUS + origin.y);
		ImVec2 weight_pt(w_x, w_y);
		//circle(canvas.canvas, weight_pt, weight_radius, black, 1, LINE_8);
		//putText(canvas.canvas, to_string(i), weight_pt, FONT_HERSHEY_COMPLEX_SMALL, 1.0, BLACK);
	}*/

	// Draw the bias
	double bx = 0;
	double by = 0;
	if (activationFunction->hasBias())
	{
		bx = origin.x + (BIAS_OFFSET_X * scale);
		by = origin.y + (BIAS_OFFSET_Y * scale);
		ImVec2 bTL = ImVec2(bx, by);
		ImVec2 bBR = ImVec2(bx + (BIAS_WIDTH * scale), by + (BIAS_HEIGHT * scale));
		ImVec2 biasPt = ImVec2(bx + (BIAS_TEXT_X * scale), by);
		canvas->AddRect(bTL, bBR, BLACK);
		canvas->AddText(ImGui::GetFont(), (BIAS_FONT_SIZE * scale), biasPt, BLACK, to_string(1).c_str());
	}
	else { }

	// Draw the links
	ImVec2 pt = ImVec2(origin);
	pt.y -= RADIUS * scale;
	// Draw the links to the previous neurons
	double previousX, previousY;
	for (int i = 0; i < inputCount; i++)
	{
		if (parentCount == 0 && i == 0)
		{
			previousX = pt.x;
			previousY = pt.y - LINE_LENGTH;
		}
		else if (i < (inputCount - 1) || !activationFunction->hasBias())
		{
			previousX = parents->at(i)->position.x;
			previousY = parents->at(i)->position.y + (RADIUS * scale);
		}
		else
		{
			previousX = bx;
			previousY = by + (BIAS_HEIGHT / 2.0);
		}

		ImVec2 previousNeuronPt(previousX, previousY);
		ImColor lineColor = ImColor(1.0f, 1.0f, 1.0f, 1.0f);
		int lineWidth = 1;
		float weight = activationFunction->getWeights().getParameters()(i);
		if (weight >= 0.0f)
		{
			if (weight <= 1.0f)
			{
				lineColor = ImColor(1.0f - weight, 1.0f - weight, 1.0f - weight, 1.0f);
				lineWidth = ((int)(ceil(weight * lineWidth)));
			}
			else 
			{
				lineColor = ImColor(0.0f, 0.0f, 0.0f, 1.0f);
				lineWidth = ((int)(ceil(1.0f * lineWidth)));
			}
		}
		else
		{
			if (weight >= -1.0f)
			{
				lineColor = ImColor(-weight, 0.0f, 0.0f, 1.0f);
				lineWidth = ((int)(ceil(-weight * lineWidth)));
			}
			else
			{
				lineColor = ImColor(1.0f, 0.0f, 0.0f, 1.0f);
				lineWidth = ((int)(ceil(1.0f * lineWidth)));
			}
		}
		if (lineWidth > 0)
		{
			canvas->AddLine(previousNeuronPt, pt, lineColor, lineWidth);
		}
		else
		{
			canvas->AddLine(previousNeuronPt, pt, WHITE, 1);
		}
	}

	if (output)
	{
		// Draw the output lines
		pt.y += (2 * RADIUS * scale);
		ImVec2 next(pt.x, pt.y + (LINE_LENGTH * scale));
		canvas->AddLine(next, pt, GRAY, 1);
	}
	else { }
}