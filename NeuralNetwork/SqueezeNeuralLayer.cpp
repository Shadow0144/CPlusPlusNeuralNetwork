#define _USE_MATH_DEFINES

#include "SqueezeNeuralLayer.h"

#include <math.h>
#include <tuple>

// Input shape is the shape of a single example
SqueezeNeuralLayer::SqueezeNeuralLayer(std::vector<size_t> squeezeDims)
{
	this->children = NULL;
	this->squeezeDims = squeezeDims;
	this->numUnits = 1;
}

SqueezeNeuralLayer::~SqueezeNeuralLayer()
{

}

void SqueezeNeuralLayer::addChildren(NeuralLayer* children)
{
	this->children = children;
}

xt::xarray<double> SqueezeNeuralLayer::feedForward(xt::xarray<double> input)
{
	return input;
}

xt::xarray<double> SqueezeNeuralLayer::backPropagate(xt::xarray<double> sigmas)
{
	return sigmas;
}

double SqueezeNeuralLayer::applyBackPropagate()
{
	return 0;
}

std::vector<size_t> SqueezeNeuralLayer::getOutputShape()
{
	return std::vector<size_t>();
}

void SqueezeNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor VERY_LIGHT_GRAY(0.8f, 0.8f, 0.8f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);
	const double LINE_LENGTH = 15;

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