#pragma once

#pragma warning(push, 0)
#include "imgui.h"
#include <xtensor/xarray.hpp>
#pragma warning(pop)

enum class ActivationFunction
{
	Linear,
	ReLU,
	AbsoluteReLU,
	ELU,
	SELU,
	GELU,
	LeakyReLU,
	PReLU,
	ReLU6,
	ReLUn,
	Softplus,
	Exponential,
	Quadratic,
	Sigmoid,
	Tanh,
	HardSigmoid,
	Softsign,
	Swish
};

class NeuralLayer
{
public:
	virtual void addChildren(NeuralLayer* children) = 0;
	size_t getNumUnits() { return numUnits; }

	virtual xt::xarray<double> feedForward(xt::xarray<double> input) = 0;
	virtual xt::xarray<double> backPropagate(xt::xarray<double> sigmas) = 0;
	virtual double applyBackPropagate() = 0;

	virtual std::vector<size_t> getOutputShape() = 0;

	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output) = 0;

	// Drawing constants
	const static double RADIUS;// = 40;
	const static double DIAMETER;// = RADIUS * 2;
	const static double NEURON_SPACING;// = 20;
	static double getLayerWidth(size_t numUnits, double scale); // Drawing helper function
	static double getNeuronX(double originX, double layerWidth, int i, double scale); // Drawing helper function

protected:
	size_t numUnits;
};