#pragma once

#include "NeuralLayer/NeuralLayer.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#include <shared_mutex>
#pragma warning(pop)

class SoftmaxNeuralLayer : public NeuralLayer
{
public:
	SoftmaxNeuralLayer(NeuralLayer* parent, int axis = -1);
	~SoftmaxNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void useSimplifiedGradient(bool useOptimizedGradient); // For use by the cross entropy loss function for optimized gradient calculations
	bool isSoftmaxLayer(); // For use by the cross entropy loss function for optimized gradient calculations
	
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	size_t numInputs;
	size_t numOutputs;
	int axis;

	mutable std::shared_mutex outputMutex;

	std::vector<int> sumIndices;

	bool useOptimizedGradient; // Use the more optimized gradient calculation when the final layer and the loss function is cross entropy

	// Standard case where not the final layer or the error function is not cross entropy
	xt::xarray<double> getGradientStandard(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	// Special case where the final layer and the error function is cross entropy
	xt::xarray<double> getGradientCrossEntropy(const xt::xarray<double>& sigmas, Optimizer* optimizer);

	void drawSoftmax(ImDrawList* canvas, ImVec2 origin, double scale);
};