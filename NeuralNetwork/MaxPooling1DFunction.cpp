#include "MaxPooling1DFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#pragma warning(pop)

using namespace std;

// TODO: Padding and dimensions
MaxPooling1DFunction::MaxPooling1DFunction(const std::vector<size_t>& filterShape)
{
	this->hasBias = false;
	this->filterShape = filterShape;
}

xt::xarray<double> MaxPooling1DFunction::feedForward(const xt::xarray<double>& inputs)
{
	const int DIM1 = inputs.dimension() - 2; // First dimension
	const int DIMC = inputs.dimension() - 1; // Channels
	auto shape = inputs.shape();
	shape[DIM1] = ceil((shape[DIM1] - (filterShape[0] - 1)) / filterShape[0]);
	xt::xarray<double> output = xt::xarray<double>(shape);

	xt::xstrided_slice_vector inputWindowView;
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f <= DIMC; f++)
	{
		inputWindowView.push_back(xt::all());
		outputWindowView.push_back(xt::all());
	}

	int j = 0;
	const int I = (inputs.shape()[DIM1] - filterShape[0] + 1);
	for (int i = 0; i < I; i += filterShape[0])
	{
		inputWindowView[DIM1] = xt::range(i, i + filterShape[0]);
		outputWindowView[DIM1] = j++; // Increment after assignment
		auto window = xt::xarray<double>(xt::strided_view(inputs, inputWindowView));
		xt::strided_view(output, outputWindowView) = xt::amax(window, { DIM1 });
	}

	return output;
}

xt::xarray<double> MaxPooling1DFunction::backPropagate(const xt::xarray<double>& sigmas)
{
	xt::xarray<double> sigmasPrime = xt::where(xt::equal(lastInput, lastOutput), 1, 0) * sigmas;
	return sigmasPrime;
}

void MaxPooling1DFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	xt::xarray<double> drawWeights = weights.getParameters();

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = drawWeights(0, i);
		double inv_slope = 1.0 / abs(slope);
		double x1, x2, y1, y2;
		if (slope > 0.0)
		{
			x1 = -1.0;
			x2 = +min(1.0, inv_slope);
			y1 = 0.0;
			y2 = (x2 * slope);
		}
		else
		{
			x1 = -min(1.0, inv_slope);
			x2 = 1.0;
			y1 = (x1 * slope);
			y2 = 0.0;
		}

		ImVec2 l_start(position.x + (DRAW_LEN * x1 * scale), position.y - (DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (DRAW_LEN * x2 * scale), position.y - (DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}