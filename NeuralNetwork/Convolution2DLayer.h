#pragma once

#include "NeuralLayer.h"
#include "Function.h"
#include "imgui.h"
#include <vector>

using namespace std;

class Convolution2DLayer : public NeuralLayer
{
public:
	Convolution2DLayer(NeuralLayer* parent, size_t numFilters, std::vector<size_t> convolutionShape, size_t stride = 1);
	~Convolution2DLayer();

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> errors);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	Function* convolution2DFunction;
	NeuralLayer* parent;
	NeuralLayer* children;
	std::vector<size_t> inputShape;

	void addChildren(NeuralLayer* children);
};