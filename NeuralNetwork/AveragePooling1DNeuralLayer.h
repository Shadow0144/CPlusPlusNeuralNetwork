#pragma once

#pragma warning(pop)
#include "NeuralLayer.h"
#include "Function.h"
#include "imgui.h"
#include <vector>
#pragma warning(pop)

class AveragePooling1DNeuralLayer : public NeuralLayer
{
public:
	AveragePooling1DNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape);
	~AveragePooling1DNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	NeuralLayer* parent;
	NeuralLayer* children;
	std::vector<size_t> inputShape;
	std::vector<size_t> filterShape;
	xt::xarray<double> lastInput;
	xt::xarray<double> lastOutput;

	ParameterSet weights;
	const double ALPHA = 0.001; // Learning rate

	void addChildren(NeuralLayer* children);
	void draw1DPooling(ImDrawList* canvas, ImVec2 origin, double scale);
};