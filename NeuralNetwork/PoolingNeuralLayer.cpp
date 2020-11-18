#include "PoolingNeuralLayer.h"

#include "MaxPooling1DFunction.h"
#include "MaxPooling2DFunction.h"
#include "MaxPooling3DFunction.h"
#include "AveragePooling1DFunction.h"
#include "AveragePooling2DFunction.h"
#include "AveragePooling3DFunction.h"

#include <math.h>
#include <tuple>

PoolingNeuralLayer::PoolingNeuralLayer(PoolingActivationFunction function, NeuralLayer* parent, std::vector<size_t> filterShape)
{
	this->parent = parent;
	this->children = NULL;
	if (parent != NULL)
	{
		parent->addChildren(this);
	}
	else { }
	this->numUnits = 1;
	functionType = function;
	switch (functionType)
	{
		case PoolingActivationFunction::Max1D:
			activationFunction = new MaxPooling1DFunction(filterShape);
			break;
		case PoolingActivationFunction::Max2D:
			activationFunction = new MaxPooling2DFunction(filterShape);
			break;
		case PoolingActivationFunction::Max3D:
			activationFunction = new MaxPooling3DFunction(filterShape);
			break;
		case PoolingActivationFunction::Average1D:
			activationFunction = new AveragePooling1DFunction(filterShape);
			break;
		case PoolingActivationFunction::Average2D:
			activationFunction = new AveragePooling2DFunction(filterShape);
			break;
		case PoolingActivationFunction::Average3D:
			activationFunction = new AveragePooling3DFunction(filterShape);
			break;
		default:
			activationFunction = new MaxPooling1DFunction(filterShape);
			break;
	}
}

PoolingNeuralLayer::~PoolingNeuralLayer()
{
	delete activationFunction;
}

void PoolingNeuralLayer::addChildren(NeuralLayer* children)
{
	this->children = children;
}

xt::xarray<double> PoolingNeuralLayer::feedForward(xt::xarray<double> input)
{
	return activationFunction->feedForward(input);
}

xt::xarray<double> PoolingNeuralLayer::feedForwardTrain(xt::xarray<double> input)
{
	return activationFunction->feedForwardTrain(input);
}

xt::xarray<double> PoolingNeuralLayer::backPropagate(xt::xarray<double> sigmas)
{
	return activationFunction->backPropagate(sigmas);
}

double PoolingNeuralLayer::applyBackPropagate()
{
	return activationFunction->applyBackPropagate();
}

std::vector<size_t> PoolingNeuralLayer::getOutputShape()
{
	return activationFunction->getOutputShape();
}

void PoolingNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor VERY_LIGHT_GRAY(0.8f, 0.8f, 0.8f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);
	const double LINE_LENGTH = 15;

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

	// Draw each neuron
	for (int i = 0; i < numUnits; i++)
	{
		currentNeuronPt.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		for (int j = 0; j < parentCount; j++) // There should be at least one parent
		{
			previousX = NeuralLayer::getNeuronX(origin.x, PARENT_LAYER_WIDTH, j, scale);
			ImVec2 previousNeuronPt(previousX, previousY);

			// Draw line to previous neuron
			canvas->AddLine(previousNeuronPt, currentNeuronPt, BLACK, 1.0f);
		}
	}

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