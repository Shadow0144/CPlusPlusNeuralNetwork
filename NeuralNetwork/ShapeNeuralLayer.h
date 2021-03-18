#pragma once

#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <vector>
#pragma warning(pop)

class ShapeNeuralLayer : public NeuralLayer
{
public:
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

protected:
	xt::svector<size_t> lastShape; // The shape of the last training input
};