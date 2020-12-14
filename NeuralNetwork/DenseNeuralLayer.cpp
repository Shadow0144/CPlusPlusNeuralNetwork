#define _USE_MATH_DEFINES

#include "DenseNeuralLayer.h"
#include "LinearFunction.h"
#include "ReLUFunction.h"
#include "AbsoluteReLUFunction.h"
#include "CReLUFunction.h"
#include "ELUFunction.h"
#include "SELUFunction.h"
#include "GELUFunction.h"
#include "LeakyReLUFunction.h"
#include "PReLUFunction.h"
#include "ReLU6Function.h"
#include "ReLUnFunction.h"
#include "SoftplusFunction.h"
#include "ExponentialFunction.h"
#include "QuadraticFunction.h"
#include "SigmoidFunction.h"
#include "TanhFunction.h"
#include "HardSigmoidFunction.h"
#include "SoftsignFunction.h"
#include "SwishFunction.h"
#include "MaxoutFunction.h"

#pragma warning(push, 0)
#include <math.h>
#include <tuple>
#pragma warning(pop)

DenseNeuralLayer::DenseNeuralLayer(DenseActivationFunction function, NeuralLayer* parent, size_t numUnits)
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
		case DenseActivationFunction::Linear:
			activationFunction = new LinearFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::ReLU:
			activationFunction = new ReLUFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::AbsoluteReLU:
			activationFunction = new AbsoluteReLUFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::CReLU:
			activationFunction = new CReLUFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::ELU:
			activationFunction = new ELUFunction(parent->getNumUnits(), numUnits);
			break;		
		case DenseActivationFunction::SELU:
			activationFunction = new SELUFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::GELU:
			activationFunction = new GELUFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::LeakyReLU:
			activationFunction = new LeakyReLUFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::PReLU:
			activationFunction = new PReLUFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::ReLU6:
			activationFunction = new ReLU6Function(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::ReLUn:
			activationFunction = new ReLUnFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::Softplus:
			activationFunction = new SoftplusFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::Exponential:
			activationFunction = new ExponentialFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::Quadratic:
			activationFunction = new QuadraticFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::Sigmoid:
			activationFunction = new SigmoidFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::Tanh:
			activationFunction = new TanhFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::HardSigmoid:
			activationFunction = new HardSigmoidFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::Softsign:
			activationFunction = new SoftsignFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::Swish:
			activationFunction = new SwishFunction(parent->getNumUnits(), numUnits);
			break;
		case DenseActivationFunction::Maxout:
			activationFunction = new MaxoutFunction(parent->getNumUnits(), numUnits, 5); // TODO
			break;
		default:
			activationFunction = new LinearFunction(parent->getNumUnits(), numUnits);
			break;
	}
}

DenseNeuralLayer::~DenseNeuralLayer()
{
	delete activationFunction;
}

xt::xarray<double> DenseNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	xt::xarray<double> output;
	if (activationFunction->getHasBias())
	{
		output = activationFunction->feedForward(activationFunction->addBias(input));
	}
	else 
	{
		output = activationFunction->feedForward(input);
	}
	return output;
}

xt::xarray<double> DenseNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	xt::xarray<double> output;
	if (activationFunction->getHasBias())
	{
		output = activationFunction->feedForwardTrain(activationFunction->addBias(input));
	}
	else
	{
		output = activationFunction->feedForwardTrain(input);
	}
	return output;
}

xt::xarray<double> DenseNeuralLayer::backPropagate(const xt::xarray<double>& sigmas)
{
	return activationFunction->backPropagate(sigmas);
}

double DenseNeuralLayer::applyBackPropagate()
{
	return activationFunction->applyBackPropagate();
}

std::vector<size_t> DenseNeuralLayer::getOutputShape()
{
	return activationFunction->getOutputShape();
}

void DenseNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	// Draw the neurons
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);
	}

	// Draw the activation function
	activationFunction->draw(canvas, origin, scale);

	// Draw the links to the previous neurons
	double previousX, previousY;
	int parentCount = parent->getNumUnits();
	const double PARENT_LAYER_WIDTH = NeuralLayer::getLayerWidth(parentCount, scale);
	ImVec2 currentNeuronPt(0, origin.y - (RADIUS * scale));
	previousY = origin.y - (DIAMETER * scale);

	xt::xarray<double> weights = activationFunction->getWeights().getParameters();

	// Set up bias parameters
	double biasX = NeuralLayer::getNeuronX(origin.x, PARENT_LAYER_WIDTH, parentCount, scale);
	double biasY = previousY - RADIUS * scale;
	ImVec2 biasPt(biasX + 0.5 * (BIAS_WIDTH * scale), biasY + (BIAS_HEIGHT * scale));

	// Draw each neuron
	for (int i = 0; i < numUnits; i++)
	{
		currentNeuronPt.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		for (int j = 0; j < parentCount; j++) // There should be at least one parent
		{
			previousX = NeuralLayer::getNeuronX(origin.x, PARENT_LAYER_WIDTH, j, scale);
			ImVec2 previousNeuronPt(previousX, previousY);

			// Decide line color and width
			ImColor lineColor = ImColor(1.0f, 1.0f, 1.0f, 1.0f);
			float lineWidth = (1.0f / 36.0f) * scale;
			float weight = weights(i);
			if (weight >= 0.0f)
			{
				if (weight <= 1.0f)
				{
					lineColor = ImColor(1.0f - weight, 1.0f - weight, 1.0f - weight, 1.0f);
					lineWidth = weight * lineWidth;
				}
				else
				{
					lineColor = ImColor(0.0f, 0.0f, 0.0f, 1.0f);
					lineWidth = 1.0f * lineWidth;
				}
			}
			else
			{
				if (weight >= -1.0f)
				{
					lineColor = ImColor(-weight, 0.0f, 0.0f, 1.0f);
					lineWidth = -weight * lineWidth;
				}
				else
				{
					lineColor = ImColor(1.0f, 0.0f, 0.0f, 1.0f);
					lineWidth = 1.0f * lineWidth;
				}
			}

			// Draw line
			if (lineWidth > 0)
			{
				canvas->AddLine(previousNeuronPt, currentNeuronPt, lineColor, lineWidth);
			}
			else
			{
				canvas->AddLine(previousNeuronPt, currentNeuronPt, WHITE, 1.0f);
			}
		}

		// Consider moving to another function for second pass
		if (activationFunction->getHasBias())
		{
			// Draw the bias line
			ImColor lineColor = ImColor(1.0f, 1.0f, 1.0f, 1.0f);
			float lineWidth = (1.0f / 36.0f) * scale;
			float weight = ((float)(weights(numUnits - 1)));
			if (weight >= 0.0f)
			{
				if (weight <= 1.0f)
				{
					lineColor = ImColor(1.0f - weight, 1.0f - weight, 1.0f - weight, 1.0f);
					lineWidth = weight * lineWidth;
				}
				else
				{
					lineColor = ImColor(0.0f, 0.0f, 0.0f, 1.0f);
					lineWidth = 1.0f * lineWidth;
				}
			}
			else
			{
				if (weight >= -1.0f)
				{
					lineColor = ImColor(-weight, 0.0f, 0.0f, 1.0f);
					lineWidth = -weight * lineWidth;
				}
				else
				{
					lineColor = ImColor(1.0f, 0.0f, 0.0f, 1.0f);
					lineWidth = 1.0f * lineWidth;
				}
			}
			if (lineWidth > 0)
			{
				canvas->AddLine(biasPt, currentNeuronPt, lineColor, lineWidth);
			}
			else
			{
				canvas->AddLine(biasPt, currentNeuronPt, WHITE, 1.0f);
			}
		}
		else { }
	} // for (int i = 0; i < numUnits; i++)

	if (activationFunction->getHasBias())
	{
		// Draw the bias box
		ImVec2 bTL = ImVec2(biasX, biasY);
		ImVec2 bBR = ImVec2(biasX + (BIAS_WIDTH * scale), biasY + (BIAS_HEIGHT * scale));
		biasPt = ImVec2(biasX + (BIAS_TEXT_X * scale), biasY);
		canvas->AddRectFilled(bTL, bBR, VERY_LIGHT_GRAY);
		canvas->AddRect(bTL, bBR, BLACK);
		canvas->AddText(ImGui::GetFont(), (BIAS_FONT_SIZE * scale), biasPt, BLACK, to_string(1).c_str());
	}
	else { }

	if (output)
	{
		for (int i = 0; i < numUnits; i++)
		{
			// Draw the output lines
			double x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
			ImVec2 outputPt(x, position.y + (RADIUS * scale));
			ImVec2 nextPt(x, outputPt.y + (LINE_LENGTH * scale));
			canvas->AddLine(outputPt, nextPt, GRAY);
		}
	}
	else { }

	// Overlaying black ring
	for (int i = 0; i < numUnits; i++)
	{
		position.x = origin.x - (LAYER_WIDTH * 0.5) + (((DIAMETER + NEURON_SPACING) * i) * scale);
		canvas->AddCircle(position, RADIUS * scale, BLACK, 32);
	}
}