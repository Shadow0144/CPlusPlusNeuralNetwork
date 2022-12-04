#pragma once

#include "NeuralLayer/NeuralLayer.h"

#pragma warning(push, 0)
#include <vector>
#pragma warning(pop)

class ShapeNeuralLayer : public NeuralLayer
{
public:
	virtual ~ShapeNeuralLayer() = 0; // This class is not intended to be directly instantiated

	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

protected:
	ShapeNeuralLayer(NeuralLayer* parent);

	xt::svector<size_t> lastShape; // The shape of the last training input
};