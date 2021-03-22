#pragma once

#include "NeuralLayer.h"
#include "ActivationFunction.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#include <map>
#pragma warning(pop)

using namespace std;

class ActivationFunctionNeuralLayer : public NeuralLayer
{
public:
	ActivationFunctionNeuralLayer(ActivationFunctionType functionType, NeuralLayer* parent,
		std::map<string, double> additionalParameters = std::map<string, double>());
	~ActivationFunctionNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void saveParameters(std::string fileName);
	void loadParameters(std::string fileName);

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	ActivationFunctionType functionType;
	ActivationFunction* activationFunction;
	int numInputs;
};