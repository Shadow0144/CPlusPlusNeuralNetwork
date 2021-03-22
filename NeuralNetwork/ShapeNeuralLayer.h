#pragma once

#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <vector>
#pragma warning(pop)

class ShapeNeuralLayer : public NeuralLayer
{
public:
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

protected:
	xt::svector<size_t> lastShape; // The shape of the last training input
};