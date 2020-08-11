#define _USE_MATH_DEFINES

#include "InputNeuralLayer.h"

#include <math.h>
#include <tuple>

// Input shape is the shape of a single example
InputNeuralLayer::InputNeuralLayer(std::vector<size_t> inputShape)
{
	this->children = NULL;
	this->inputShape = inputShape;
	this->numUnits = inputShape.at(inputShape.size() - 1);
}

InputNeuralLayer::~InputNeuralLayer()
{

}

void InputNeuralLayer::addChildren(NeuralLayer* children)
{
	this->children = children;
}

xt::xarray<double> InputNeuralLayer::feedForward(xt::xarray<double> input)
{
	return input;
}

xt::xarray<double> InputNeuralLayer::backPropagate(xt::xarray<double> errors)
{
	return errors;
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
	
}