#pragma once

#pragma warning(push, 0)
#include "imgui.h"
#include <xtensor/xarray.hpp>
#pragma warning(pop)

enum class ActivationFunction
{
	Identity,
	WeightedDotProduct,
	ReLU,
	LeakyReLU,
	Softplus,
	Sigmoid,
	Tanh,
	Softmax,
	Convolution,
	Max
};

class NeuralLayer
{
public:
	virtual void addChildren(NeuralLayer* children) = 0;
	size_t getNumUnits() { return numUnits; }

	virtual xt::xarray<double> feedForward(xt::xarray<double> input) = 0;
	virtual xt::xarray<double> backPropagate(xt::xarray<double> errors) = 0;
	virtual double applyBackPropagate() = 0;

	virtual std::vector<size_t> getOutputShape() = 0;

	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output) = 0;

protected:
	size_t numUnits;
};