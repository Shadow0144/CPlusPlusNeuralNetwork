#define _USE_MATH_DEFINES

#include "ActivationFunctionNeuralLayer.h"

#include "ActivationFunctionFactory.h"

#pragma warning(push, 0)
#include <math.h>
#include <tuple>
#pragma warning(pop)

#include "Test.h"

ActivationFunctionNeuralLayer::ActivationFunctionNeuralLayer(ActivationFunctionType functionType, NeuralLayer* parent, std::map<string, double> additionalParameters)
{
	this->numInputs = 0;
	this->parent = parent;
	this->children = nullptr;
	if (parent != nullptr)
	{
		parent->addChildren(this);
		this->numInputs = parent->getNumUnits();
	}
	else { }
	this->numUnits = 1;
	this->functionType = functionType;
	this->activationFunction = ActivationFunctionFactory::getNewActivationFunction(functionType, additionalParameters);
}

ActivationFunctionNeuralLayer::~ActivationFunctionNeuralLayer()
{
	delete activationFunction;
}

xt::xarray<double> ActivationFunctionNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	// Apply the activation function
	return activationFunction->feedForward(input);
}

xt::xarray<double> ActivationFunctionNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	return activationFunction->feedForwardTrain(input);
}

xt::xarray<double> ActivationFunctionNeuralLayer::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return activationFunction->getGradient(sigmas, optimizer);
}

double ActivationFunctionNeuralLayer::applyBackPropagate()
{
	activationFunction->applyBackPropagate(); // Update any parameters the activation function needs to change
	return 0; // No weights to update
}

std::vector<size_t> ActivationFunctionNeuralLayer::getOutputShape()
{
	std::vector<size_t> outputShape;
	outputShape.push_back(numUnits);
	outputShape = activationFunction->getOutputShape(outputShape);
	return outputShape;
}

void ActivationFunctionNeuralLayer::saveParameters(std::string fileName)
{
	activationFunction->saveParameters(fileName);
}

void ActivationFunctionNeuralLayer::loadParameters(std::string fileName)
{
	activationFunction->loadParameters(fileName);
}

void ActivationFunctionNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	// Draw the neurons
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);
	}

	// Draw the activation function (If the activation function type is Identity, the Linear function will still draw)
	ParameterSet weights = ParameterSet();
	weights.setParametersOne({ 1 });
	activationFunction->draw(canvas, origin, scale, numUnits, weights);

	// Draw the links to the previous neurons
	double previousX, previousY;
	int parentCount = parent->getNumUnits();
	const double PARENT_LAYER_WIDTH = NeuralLayer::getLayerWidth(parentCount, scale);
	ImVec2 currentNeuronPt(0, origin.y - (RADIUS * scale));
	previousY = origin.y - (DIAMETER * scale);

	xt::xarray<double> drawWeights = weights.getParameters();

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
			float weight = drawWeights(i);
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
	} // for (int i = 0; i < numUnits; i++)

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