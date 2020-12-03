#include "Convolution3DFunction.h"

#pragma warning(push, 0)
#include <xtensor/xview.hpp>
#pragma warning(pop)

Convolution3DFunction::Convolution3DFunction(const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride, size_t numKernels)
{
	this->hasBias = false;
	this->numUnits = 1;
	this->convolutionShape = convolutionShape;
	this->stride = stride;
	this->inputChannels = inputChannels;
	this->numKernels = numKernels;
	this->drawAxes = false;

	std::vector<size_t> paramShape;
	// convolution x ... x filters x kernel -shaped
	for (int i = 0; i < convolutionShape.size(); i++)
	{
		paramShape.push_back(convolutionShape[i]);
	}
	paramShape.push_back(inputChannels);
	paramShape.push_back(numKernels);
	this->weights.setParametersRandom(paramShape);

	kernelWindowView = xt::xstrided_slice_vector();
	kernelWindowView.push_back(xt::all()); // First dimension
	kernelWindowView.push_back(xt::all()); // Second dimension
	kernelWindowView.push_back(xt::all()); // Third dimension
	kernelWindowView.push_back(xt::all()); // Channels / Filters
	kernelWindowView.push_back(0); // Current kernel
}

xt::xarray<double> Convolution3DFunction::feedForward(const xt::xarray<double>& inputs)
{
	auto inputShape = inputs.shape();
	size_t stopi = inputShape[0] - convolutionShape[0];
	size_t stopj = inputShape[1] - convolutionShape[1];
	xt::xarray<double>::shape_type shape = { stopi + 1, stopj + 1 };
	xt::xarray<double> output(shape);

	for (size_t i = 0; i < stopi; i++)
	{
		for (size_t j = 0; j < stopj; j++)
		{
			auto block = xt::view(inputs, (i, j, convolutionShape[0], convolutionShape[1]));
			auto prod = block * weights.getParameters();
			output(i, j) = xt::sum(prod)();
		}
	}

	return output;
}

xt::xarray<double> Convolution3DFunction::backPropagate(const xt::xarray<double>& sigmas)
{
	//weights.incrementDeltaParameters(-ALPHA * lastInput.transpose() * 0.0);
	return sigmas;
}

void Convolution3DFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	/*const Scalar BLACK(0, 0, 0);

	Point l_start(canvas.offset.x - DRAW_LEN, canvas.offset.y - ((int)(-DRAW_LEN)));
	Point l_end(canvas.offset.x + DRAW_LEN, canvas.offset.y - ((int)(DRAW_LEN)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);*/
}