#pragma once

#pragma warning(push, 0)
#include "NeuralLayer.h"
#include "Function.h"
#include "imgui.h"
#include <vector>
#pragma warning(pop)

using namespace std;

class DenseNeuralLayer : public NeuralLayer
{
public:
	DenseNeuralLayer(DenseActivationFunction function, NeuralLayer* parent, size_t numUnits);
	~DenseNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	DenseActivationFunction functionType;
	Function* activationFunction;
};