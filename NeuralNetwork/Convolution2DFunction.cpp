#include "Convolution2DFunction.h"

#pragma warning(push, 0)
#include <xtensor/xview.hpp>
#pragma warning(pop)

Convolution2DFunction::Convolution2DFunction(size_t incomingUnits, size_t numFilters, std::vector<size_t> convolutionShape, int stride)
{
	this->hasBias = false;
	this->numInputs = incomingUnits;
	this->numUnits = 1;
	this->numFilters = numFilters;
	this->convolutionShape = convolutionShape;
	this->stride = stride;
	this->drawAxes = false;
}

xt::xarray<double> Convolution2DFunction::feedForward(xt::xarray<double> input)
{
	lastInput = input;
	auto inputShape = input.shape();
	size_t stopi = inputShape[0] - convolutionShape[0];
	size_t stopj = inputShape[1] - convolutionShape[1];
	xt::xarray<double>::shape_type shape = { stopi + 1, stopj + 1 };
	xt::xarray<double> lastOutput(shape);

	for (size_t i = 0; i < stopi; i++)
	{
		for (size_t j = 0; j < stopj; j++)
		{
			auto block = xt::view(input, (i, j, convolutionShape[0], convolutionShape[1]));
			auto prod = block * weights.getParameters();
			lastOutput(i, j) = xt::sum(prod)();
		}
	}

	return lastOutput;
}

xt::xarray<double> Convolution2DFunction::backPropagate(xt::xarray<double> sigmas)
{
	//weights.incrementDeltaParameters(-ALPHA * lastInput.transpose() * 0.0);
	return sigmas;
}

void Convolution2DFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	/*const Scalar BLACK(0, 0, 0);

	Point l_start(canvas.offset.x - DRAW_LEN, canvas.offset.y - ((int)(-DRAW_LEN)));
	Point l_end(canvas.offset.x + DRAW_LEN, canvas.offset.y - ((int)(DRAW_LEN)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);*/
}