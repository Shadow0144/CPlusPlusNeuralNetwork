#pragma once

#include "NeuralLayer/ParameterizedNeuralLayer.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#include <map>
#pragma warning(pop)

using namespace std;

class DenseNeuralLayer : public ParameterizedNeuralLayer
{
public:
	DenseNeuralLayer(ActivationFunctionType functionType, NeuralLayer* parent, size_t numUnits, 
		std::map<string, double> additionalParameters = std::map<string, double>(), bool addBias = true);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	xt::xarray<double> lastInput;
	xt::xarray<double> lastOutput;
	bool addBias;
	int numInputs;

	xt::xarray<double> dotProduct(const xt::xarray<double>& input);
	xt::xarray<double> denseBackpropagate(const xt::xarray<double>& sigmas, Optimizer* optimizer);
};