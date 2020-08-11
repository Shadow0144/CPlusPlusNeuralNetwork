#define _USE_MATH_DEFINES

#include "DenseNeuralLayer.h"
#include "IdentityFunction.h"
#include "DotProductFunction.h"
#include "ReLUFunction.h"
#include "LeakyReLUFunction.h"
#include "SoftplusFunction.h"
#include "SigmoidFunction.h"
#include "TanhFunction.h"
#include "SoftmaxFunction.h"
#include "ConvolutionFunction.h"

#include <math.h>
#include <tuple>

DenseNeuralLayer::DenseNeuralLayer(ActivationFunction function, NeuralLayer* parent, size_t numUnits)
{
	this->parent = parent;
	this->children = NULL;
	if (parent != NULL)
	{
		parent->addChildren(this);
	}
	else { }
	this->numUnits = numUnits;
	functionType = function;
	switch (functionType)
	{
		case ActivationFunction::Identity:
			//activationFunction = new IdentityFunction();
			break;
		case ActivationFunction::WeightedDotProduct:
			activationFunction = new DotProductFunction(parent->getNumUnits(), numUnits);
			break;
		case ActivationFunction::ReLU:
			activationFunction = new ReLUFunction(parent->getNumUnits(), numUnits);
			break;
		case ActivationFunction::LeakyReLU:
			activationFunction = new LeakyReLUFunction(parent->getNumUnits(), numUnits);
			break;
		case ActivationFunction::Softplus:
			activationFunction = new SoftplusFunction(parent->getNumUnits(), numUnits);
			break;
		case ActivationFunction::Sigmoid:
			activationFunction = new SigmoidFunction(parent->getNumUnits(), numUnits);
			break;
		case ActivationFunction::Tanh:
			activationFunction = new TanhFunction(parent->getNumUnits(), numUnits);
			break;
		default:
			//activationFunction = new IdentityFunction();
			break;
	}
}

DenseNeuralLayer::~DenseNeuralLayer()
{
	delete activationFunction;
}

void DenseNeuralLayer::addChildren(NeuralLayer* children)
{
	this->children = children;
}

xt::xarray<double> DenseNeuralLayer::feedForward(xt::xarray<double> input)
{
	return activationFunction->feedForward(input);
}

xt::xarray<double> DenseNeuralLayer::backPropagate(xt::xarray<double> errors)
{
	return activationFunction->backPropagate(errors);
}

double DenseNeuralLayer::applyBackPropagate()
{
	return activationFunction->applyBackProgate();
}

std::vector<size_t> DenseNeuralLayer::getOutputShape()
{
	std::vector<size_t> outputShape;
	outputShape.push_back(numUnits);
	return outputShape;
}

void DenseNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	//const double P = 10;
	//const double RADIUS = 40;
	//const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	//const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	//const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	//const ImColor VERY_LIGHT_GRAY(0.8f, 0.8f, 0.8f, 1.0f);
	//const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);
	//const double LINE_LENGTH = 15;
	//const double WEIGHT_RADIUS = 10;
	//const double BIAS_OFFSET_X = 40;
	//const double BIAS_OFFSET_Y = -52;
	//const double BIAS_FONT_SIZE = 24;
	//const double BIAS_WIDTH = 20;
	//const double BIAS_HEIGHT = BIAS_FONT_SIZE;
	//const double BIAS_TEXT_X = 4;
	//const double BIAS_TEXT_Y = 20;

	//position = origin;

	//// Draw the neuron
	//canvas->AddCircleFilled(ImVec2(origin.x, origin.y), RADIUS * scale, LIGHT_GRAY, 32);

	//// Draw the activation function
	//activationFunction->draw(canvas, origin, scale);

	///*// Draw the weights
	//double i_angle = 0;
	//double angle = 0;
	//if (parentCount <= 1)
	//{
	//	i_angle = (M_PI / 2.0);
	//	angle = 0;
	//}
	//else if (parentCount % 2 == 0)
	//{
	//	i_angle = (3.0 * M_PI / 4.0);
	//	angle = M_PI / ((double)(parentCount)) / 2.0;
	//}
	//else 
	//{
	//	i_angle = (3.0 * M_PI / 4.0);
	//	angle = M_PI / (((double)(parentCount)) - 1.0) / 2.0;
	//}
	//for (int i = 0; i < parentCount; i++)
	//{
	//	double w_x = (cos(i_angle - (angle * i)) * scale * RADIUS + origin.x);
	//	double w_y = (-sin(i_angle - (angle * i)) * scale * RADIUS + origin.y);
	//	ImVec2 weight_pt(w_x, w_y);
	//	//circle(canvas.canvas, weight_pt, weight_radius, black, 1, LINE_8);
	//	//putText(canvas.canvas, to_string(i), weight_pt, FONT_HERSHEY_COMPLEX_SMALL, 1.0, BLACK);
	//}*/

	//// Setup the bias location
	//double bx = origin.x + (BIAS_OFFSET_X * scale);
	//double by = origin.y + (BIAS_OFFSET_Y * scale);

	//// Draw the links
	//ImVec2 pt = ImVec2(origin);
	//pt.y -= RADIUS * scale;
	//// Draw the links to the previous neurons
	//double previousX, previousY;
	//for (int i = 0; i < parentCount || i < 1; i++)
	//{
	//	if (parentCount == 0)
	//	{
	//		previousX = pt.x;
	//		previousY = pt.y - (LINE_LENGTH * scale);
	//	}
	//	else
	//	{
	//		previousX = parents[i]->position.x;
	//		previousY = parents[i]->position.y + (RADIUS * scale);
	//	}

	//	ImVec2 previousNeuronPt(previousX, previousY);
	//	ImColor lineColor = ImColor(1.0f, 1.0f, 1.0f, 1.0f);
	//	float lineWidth = (1.0f / 36.0f) * scale;
	//	float weight = activationFunction->getWeights().getParameters()(i);
	//	if (weight >= 0.0f)
	//	{
	//		if (weight <= 1.0f)
	//		{
	//			lineColor = ImColor(1.0f - weight, 1.0f - weight, 1.0f - weight, 1.0f);
	//			lineWidth = weight * lineWidth;
	//		}
	//		else 
	//		{
	//			lineColor = ImColor(0.0f, 0.0f, 0.0f, 1.0f);
	//			lineWidth = 1.0f * lineWidth;
	//		}
	//	}
	//	else
	//	{
	//		if (weight >= -1.0f)
	//		{
	//			lineColor = ImColor(-weight, 0.0f, 0.0f, 1.0f);
	//			lineWidth = -weight * lineWidth;
	//		}
	//		else
	//		{
	//			lineColor = ImColor(1.0f, 0.0f, 0.0f, 1.0f);
	//			lineWidth = 1.0f * lineWidth;
	//		}
	//	}
	//	if (lineWidth > 0)
	//	{
	//		canvas->AddLine(previousNeuronPt, pt, lineColor, lineWidth);
	//	}
	//	else
	//	{
	//		canvas->AddLine(previousNeuronPt, pt, WHITE, 1.0f);
	//	}
	//}

	//// Consider moving to another function for second pass
	//if (activationFunction->getHasBias())
	//{
	//	// Draw the bias line
	//	previousX = bx;
	//	previousY = by + ((BIAS_HEIGHT / 2.0) * scale);
	//	ImVec2 previousNeuronPt(previousX, previousY);
	//	ImColor lineColor = ImColor(1.0f, 1.0f, 1.0f, 1.0f);
	//	float lineWidth = (1.0f / 36.0f) * scale;
	//	float weight = ((float)(activationFunction->getWeights().getParameters()(inputCount[0]-1)));
	//	if (weight >= 0.0f)
	//	{
	//		if (weight <= 1.0f)
	//		{
	//			lineColor = ImColor(1.0f - weight, 1.0f - weight, 1.0f - weight, 1.0f);
	//			lineWidth = weight * lineWidth;
	//		}
	//		else
	//		{
	//			lineColor = ImColor(0.0f, 0.0f, 0.0f, 1.0f);
	//			lineWidth = 1.0f * lineWidth;
	//		}
	//	}
	//	else
	//	{
	//		if (weight >= -1.0f)
	//		{
	//			lineColor = ImColor(-weight, 0.0f, 0.0f, 1.0f);
	//			lineWidth = -weight * lineWidth;
	//		}
	//		else
	//		{
	//			lineColor = ImColor(1.0f, 0.0f, 0.0f, 1.0f);
	//			lineWidth = 1.0f * lineWidth;
	//		}
	//	}
	//	if (lineWidth > 0)
	//	{
	//		canvas->AddLine(previousNeuronPt, pt, lineColor, lineWidth);
	//	}
	//	else
	//	{
	//		canvas->AddLine(previousNeuronPt, pt, WHITE, 1.0f);
	//	}

	//	// Draw the bias box
	//	ImVec2 bTL = ImVec2(bx, by);
	//	ImVec2 bBR = ImVec2(bx + (BIAS_WIDTH * scale), by + (BIAS_HEIGHT * scale));
	//	ImVec2 biasPt = ImVec2(bx + (BIAS_TEXT_X * scale), by);
	//	canvas->AddRectFilled(bTL, bBR, VERY_LIGHT_GRAY);
	//	canvas->AddRect(bTL, bBR, BLACK);
	//	canvas->AddText(ImGui::GetFont(), (BIAS_FONT_SIZE * scale), biasPt, BLACK, to_string(1).c_str());
	//}
	//else { }

	//if (output)
	//{
	//	// Draw the output lines
	//	pt.y += (2 * RADIUS * scale);
	//	ImVec2 next(pt.x, pt.y + (LINE_LENGTH * scale));
	//	canvas->AddLine(next, pt, GRAY, 1.0f);
	//}
	//else { }

	//canvas->AddCircle(ImVec2(origin.x, origin.y), RADIUS * scale, BLACK, 32);
}