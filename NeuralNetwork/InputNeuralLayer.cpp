#define _USE_MATH_DEFINES

#include "InputNeuralLayer.h"

#pragma warning(push, 0)
#include <math.h>
#include <tuple>
#pragma warning(pop)

// Input shape is the shape of a single example
InputNeuralLayer::InputNeuralLayer(const std::vector<size_t>& inputShape)
{
	this->children = nullptr;
	this->inputShape = inputShape;
	this->numUnits = inputShape.at(inputShape.size() - 1);
}

InputNeuralLayer::~InputNeuralLayer()
{

}

xt::xarray<double> InputNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	return input;
}

xt::xarray<double> InputNeuralLayer::backPropagate(const xt::xarray<double>& sigmas)
{
	return sigmas;
}

double InputNeuralLayer::applyBackPropagate()
{
	return 0;
}

std::vector<size_t> InputNeuralLayer::getOutputShape()
{
	return inputShape;
}

void InputNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	// Draw the neurons
	position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddCircleFilled(position, RADIUS * scale, VERY_LIGHT_GRAY, 32);

		// Draw input line
		ImVec2 currentNeuronPt(position.x, position.y - (RADIUS * scale));
		ImVec2 previousPt(position.x, currentNeuronPt.y - (LINE_LENGTH * scale));

		canvas->AddLine(previousPt, currentNeuronPt, GRAY);

		// Overlaying black ring
		position.x = origin.x - (LAYER_WIDTH * 0.5) + (((DIAMETER + NEURON_SPACING) * i) * scale);
		canvas->AddCircle(position, RADIUS * scale, BLACK, 32);
	}
}