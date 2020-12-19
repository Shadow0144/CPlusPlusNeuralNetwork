#pragma once

#include "NeuralLayer.h"
#include "ActivationFunction.h"
#include "ActivationFunctionFactory.h"
#include "ParameterSet.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#pragma warning(pop)

using namespace std;

class DenseNeuralLayer : public NeuralLayer
{
public:
	DenseNeuralLayer(ActivationFunctionType functionType, NeuralLayer* parent, size_t numUnits, bool addBias = true);
	~DenseNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	ActivationFunctionType functionType;
	ActivationFunction* activationFunction;
	xt::xarray<double> lastInput;
	xt::xarray<double> lastOutput;
	bool addBias;
	int numInputs;

	ParameterSet weights;

	xt::xarray<double> dotProduct(const xt::xarray<double>& input);
	xt::xarray<double> denseBackpropagate(const xt::xarray<double>& sigmas);
};