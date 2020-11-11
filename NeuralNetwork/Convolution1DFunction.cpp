#include "Convolution1DFunction.h"

#pragma warning(push, 0)
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

Convolution1DFunction::Convolution1DFunction(std::vector<size_t> convolutionShape, size_t inputChannels, size_t stride, size_t numKernels)
{
	this->hasBias = false;
	this->numUnits = 1;
	this->convolutionShape = convolutionShape;
	this->stride = stride;
	this->numKernels = numKernels;
	this->drawAxes = false;

	std::vector<size_t> paramShape;
	// filter x convolution x ... x channels -shaped
	paramShape.push_back(numKernels);
	for (int i = 0; i < convolutionShape.size(); i++)
	{
		paramShape.push_back(convolutionShape[i]);
	}
	paramShape.push_back(inputChannels);
	this->weights.setParametersRandom(paramShape);
}

xt::xarray<double> Convolution1DFunction::feedForward(xt::xarray<double> inputs)
{
	lastInput = inputs;

	// Assume the last dimension is the channel dimension
	const int DIMS = inputs.dimension();
	const int DIM1 = DIMS - 2;
	const int DIMF = DIMS - 1;
	auto shape = inputs.shape();
	shape[DIM1] = ceil((shape[DIM1] - (convolutionShape[0] - 1)) / stride);
	shape[DIMF] = numKernels;
	lastOutput = xt::xarray<double>(shape);

	xt::xstrided_slice_vector kernelWindowView;
	kernelWindowView.push_back(0); // Current filter
	kernelWindowView.push_back(xt::all()); // First dimension
	kernelWindowView.push_back(xt::all()); // Channels

	xt::xstrided_slice_vector inputWindowView;
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f < DIMS; f++)
	{
		inputWindowView.push_back(xt::all());
		outputWindowView.push_back(xt::all());
	}

	for (int f = 0; f < numKernels; f++)
	{
		kernelWindowView[0] = f;
		outputWindowView[DIMF] = f;
		int j = 0;
		const int I = (inputs.shape()[DIM1] - convolutionShape[0] + 1);
		for (int i = 0; i < I; i += stride)
		{
			inputWindowView[DIM1] = xt::range(i, i + convolutionShape[0]);
			outputWindowView[DIM1] = j++; // Increment after assignment
			auto window = xt::xarray<double>(xt::strided_view(inputs, inputWindowView));
			auto filter = xt::xarray<double>(xt::strided_view(weights.getParameters(), kernelWindowView));
			xt::strided_view(lastOutput, outputWindowView) = xt::sum(window * filter);
		}
	}

	return lastOutput;
}

xt::xarray<double> Convolution1DFunction::backPropagate(xt::xarray<double> sigmas)
{
	//weights.incrementDeltaParameters(-ALPHA * lastInput.transpose() * 0.0);
	return sigmas;
}

void Convolution1DFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	/*const Scalar BLACK(0, 0, 0);

	Point l_start(canvas.offset.x - DRAW_LEN, canvas.offset.y - ((int)(-DRAW_LEN)));
	Point l_end(canvas.offset.x + DRAW_LEN, canvas.offset.y - ((int)(DRAW_LEN)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);*/
}