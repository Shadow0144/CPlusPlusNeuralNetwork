#define _USE_MATH_DEFINES

#include "NeuralLayer/DropoutNeuralLayer.h"

#pragma warning(push, 0)
#include <math.h>
#include <tuple>
#include <xtensor/xrandom.hpp>
#pragma warning(pop)

// Input shape is the shape of a single example
DropoutNeuralLayer::DropoutNeuralLayer(NeuralLayer* parent)
	: NeuralLayer(parent)
{
	this->inputShape = parent->getOutputShape();
	this->numUnits = inputShape.at(inputShape.size() - 1);
}

// Input shape is the shape of a single example
DropoutNeuralLayer::DropoutNeuralLayer(NeuralLayer* parent, double dropRate)
{
	this->parent = parent;
	this->children = nullptr;
	if (parent != nullptr)
	{
		parent->addChildren(this);
	}
	else { }
	this->inputShape = parent->getOutputShape();
	this->numUnits = inputShape.at(inputShape.size() - 1);
	if (dropRate < 0.0 || dropRate >= 1.0)
	{
		throw std::invalid_argument(std::string("Dropout rate must be in range [0, 1)"));
	}
	this->dropRate = dropRate;
}

DropoutNeuralLayer::~DropoutNeuralLayer()
{

}

xt::xarray<double> DropoutNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	return input;
}

xt::xarray<double> DropoutNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	lastMask = (xt::random::rand<double>(input.shape()) < dropRate);
	// Set dropRate% inputs to 0 and scale up all other values by (1 / (1 - dropRate))
	xt::xarray<double> output = (input * lastMask) + (input * ((1 - lastMask) * (1 / (1 - dropRate))));
	return output;
}

xt::xarray<double> DropoutNeuralLayer::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	// Sigmas need to be masked and scaled the same as the input
	xt::xarray<double> sigmasPrime = (sigmas * lastMask) + (sigmas * ((1 - lastMask) * (1 / (1 - dropRate))));
	return sigmasPrime;
}

double DropoutNeuralLayer::applyBackPropagate()
{
	return 0;
}

double DropoutNeuralLayer::getDropRate()
{
	return dropRate;
}

void DropoutNeuralLayer::setDropRate(double dropRate)
{
	if (dropRate < 0.0 || dropRate >= 1.0)
	{
		throw std::invalid_argument(std::string("Dropout rate must be in range [0, 1)"));
	}
	this->dropRate = dropRate;
}

std::vector<size_t> DropoutNeuralLayer::getOutputShape()
{
	return inputShape;
}

void DropoutNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	const double INNER_RADIUS = 24;
	const double ARROW_HALF_LENGTH = 15;
	const double ARROW_WING_WIDTH = 10;
	const double ARROW_WING_HEIGHT = 10;

	// Draw the neurons
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);

		// Draw input line
		ImVec2 currentNeuronPt(position.x, position.y - (RADIUS * scale));
		ImVec2 previousPt(position.x, origin.y - (DIAMETER * scale));

		canvas->AddLine(previousPt, currentNeuronPt, BLACK);

		position.x = origin.x - (LAYER_WIDTH * 0.5) + (((DIAMETER + NEURON_SPACING) * i) * scale);

		// Inner circle and arrow
		canvas->AddCircleFilled(position, INNER_RADIUS * scale, VERY_LIGHT_GRAY, 32);
		canvas->AddLine(ImVec2(position.x, position.y - (ARROW_HALF_LENGTH * scale)), 
			ImVec2(position.x, position.y + (ARROW_HALF_LENGTH * scale)), BLACK);
		canvas->AddLine(ImVec2(position.x, position.y + (ARROW_HALF_LENGTH * scale)),
			ImVec2(position.x - (ARROW_WING_WIDTH * scale), position.y - ((ARROW_WING_HEIGHT - ARROW_HALF_LENGTH) * scale)), BLACK);
		canvas->AddLine(ImVec2(position.x, position.y + (ARROW_HALF_LENGTH * scale)),
			ImVec2(position.x + (ARROW_WING_WIDTH * scale), position.y - ((ARROW_WING_HEIGHT - ARROW_HALF_LENGTH) * scale)), BLACK);
		canvas->AddCircle(position, INNER_RADIUS * scale, BLACK, 32);

		// Overlaying gray ring
		canvas->AddCircle(position, RADIUS * scale, BLACK, 32);
	}
}