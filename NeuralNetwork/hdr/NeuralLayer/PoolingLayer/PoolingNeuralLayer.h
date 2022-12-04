#pragma once

#include "NeuralLayer/ParameterizedNeuralLayer.h"
#include "ActivationFunction/ActivationFunction.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#pragma warning(pop)

class PoolingNeuralLayer : public ParameterizedNeuralLayer
{
public:
	PoolingNeuralLayer(NeuralLayer* parent, size_t dims, const std::vector<size_t>& filterShape, 
						const std::vector<size_t>& stride = { }, bool hasChannels = true);

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

protected:
	std::vector<size_t> filterShape;
	std::vector<size_t> stride;
	bool hasChannels;

	virtual void drawPooling(ImDrawList* canvas, ImVec2 origin, double scale) = 0;
};