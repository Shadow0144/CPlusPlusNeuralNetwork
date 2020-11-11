#define _USE_MATH_DEFINES

#include "FlattenNeuralLayer.h"

#include <math.h>
#include <tuple>

FlattenNeuralLayer::FlattenNeuralLayer(NeuralLayer* parent, int numOutputs)
{
	this->parent = parent;
	this->children = NULL;
	if (parent != NULL)
	{
		parent->addChildren(this);
	}
	else { }
	this->flattenFunction = new FlattenFunction(numOutputs);
	this->numUnits = numOutputs;
}

FlattenNeuralLayer::~FlattenNeuralLayer()
{
	delete flattenFunction;
}

void FlattenNeuralLayer::addChildren(NeuralLayer* children)
{
	this->children = children;
}

xt::xarray<double> FlattenNeuralLayer::feedForward(xt::xarray<double> input)
{
	return flattenFunction->feedForward(input);
}

xt::xarray<double> FlattenNeuralLayer::backPropagate(xt::xarray<double> sigmas)
{
	return flattenFunction->backPropagate(sigmas);
}

double FlattenNeuralLayer::applyBackPropagate()
{
	return flattenFunction->applyBackPropagate();
}

std::vector<size_t> FlattenNeuralLayer::getOutputShape()
{
	return flattenFunction->getOutputShape();
}

void FlattenNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor VERY_LIGHT_GRAY(0.8f, 0.8f, 0.8f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);
	const double LINE_LENGTH = 15;

	// Draw the neuron
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(1, scale);
	position.x = getNeuronX(origin.x, LAYER_WIDTH, 0, scale);
	canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);

	// Draw the activation function
	flattenFunction->draw(canvas, origin, scale);

	// Draw the links to the previous neurons
	double previousX, previousY;
	int parentCount = parent->getNumUnits();
	const double PARENT_LAYER_WIDTH = NeuralLayer::getLayerWidth(parentCount, scale);
	ImVec2 currentNeuronPt(0, origin.y - (RADIUS * scale));
	previousY = origin.y - (DIAMETER * scale);

	// Draw the neuron
	currentNeuronPt.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, 0, scale);
	for (int j = 0; j < parentCount; j++) // There should be at least one parent
	{
		previousX = NeuralLayer::getNeuronX(origin.x, PARENT_LAYER_WIDTH, j, scale);
		ImVec2 previousNeuronPt(previousX, previousY);

		// Draw line to previous neuron
		canvas->AddLine(previousNeuronPt, currentNeuronPt, BLACK, 1.0f);
	}

	if (output)
	{
		// Draw the output lines
		ImVec2 outputPt(position.x, position.y + (RADIUS * scale));
		ImVec2 nextPt(position.x, outputPt.y + (LINE_LENGTH * scale));
		canvas->AddLine(outputPt, nextPt, GRAY);
	}
	else { }

	// Overlaying black ring
	canvas->AddCircle(position, RADIUS * scale, BLACK, 32);
}