#include "SqueezeNeuralLayer.h"

// Input shape is the shape of a single example
SqueezeNeuralLayer::SqueezeNeuralLayer(const std::vector<size_t>& squeezeDims)
{
	this->children = NULL;
	this->squeezeDims = squeezeDims;
	this->numUnits = 1;
}

SqueezeNeuralLayer::~SqueezeNeuralLayer()
{

}

xt::xarray<double> SqueezeNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	auto shape = input.shape();
	const int DIMS = squeezeDims.size();
	for (int i = 0; i < DIMS; i++)
	{
		shape[squeezeDims[i]] = 0;
	}
	auto result = xt::xarray<double>(input);
	result.reshape(shape);
	return result;
}

xt::xarray<double> SqueezeNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	// TODO?
	return feedForward(input);
}

xt::xarray<double> SqueezeNeuralLayer::backPropagate(const xt::xarray<double>& sigmas)
{
	return sigmas; // TODO
}

double SqueezeNeuralLayer::applyBackPropagate()
{
	return 0; // No parameters
}

std::vector<size_t> SqueezeNeuralLayer::getOutputShape()
{
	return std::vector<size_t>(); // TODO
}

void SqueezeNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	// Draw the neurons
	ImVec2 position = ImVec2(origin);
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

	// TODO
}