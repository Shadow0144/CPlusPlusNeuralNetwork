#define _USE_MATH_DEFINES

#include "SoftmaxNeuralLayer.h"
#include "SoftmaxFunction.h"

#include <math.h>
#include <tuple>

SoftmaxNeuralLayer::SoftmaxNeuralLayer(NeuralLayer* parent, int axis)
{
	this->hasBias = false;
	this->parent = parent;
	this->children = NULL;
	if (parent != NULL)
	{
		parent->addChildren(this);
	}
	else { }
	this->numUnits = 1;
	this->numOutputs = parent->getNumUnits();

	this->softmaxFunction = new SoftmaxFunction(parent->getNumUnits(), axis);
}

SoftmaxNeuralLayer::~SoftmaxNeuralLayer()
{
	delete softmaxFunction;
}

void SoftmaxNeuralLayer::addChildren(NeuralLayer* children)
{
	this->children = children;
}

xt::xarray<double> SoftmaxNeuralLayer::feedForward(xt::xarray<double> input)
{
	return softmaxFunction->feedForward(input);
}

xt::xarray<double> SoftmaxNeuralLayer::backPropagate(xt::xarray<double> sigmas)
{
	return softmaxFunction->backPropagate(sigmas);
}

double SoftmaxNeuralLayer::applyBackPropagate()
{
	return softmaxFunction->applyBackPropagate();
}

std::vector<size_t> SoftmaxNeuralLayer::getOutputShape()
{
	std::vector<size_t> outputShape;
	outputShape.push_back(numUnits);
	outputShape.push_back(numOutputs);
	return outputShape;
}

void SoftmaxNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
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
			canvas->AddLine(previousNeuronPt, currentNeuronPt, GRAY, 1.0f);
		}
	} // for (int i = 0; i < numUnits; i++)

	// Draw the softmax function
	softmaxFunction->draw(canvas, origin, scale);

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