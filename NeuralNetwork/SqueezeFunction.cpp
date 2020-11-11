#include "SqueezeFunction.h"

SqueezeFunction::SqueezeFunction(std::vector<size_t> squeezeDims)
{
	this->hasBias = false;
	this->squeezeDims = squeezeDims;
}

xt::xarray<double> SqueezeFunction::feedForward(xt::xarray<double> input)
{
	auto shape = input.shape();
	const int DIMS = squeezeDims.size();
	for (int i = 0; i < DIMS; i++)
	{
		shape[squeezeDims[i]] = 0;
	}
	auto result = input.reshape(shape);
	return result;
}

xt::xarray<double> SqueezeFunction::backPropagate(xt::xarray<double> sigmas)
{
	return sigmas;
}

void SqueezeFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	ImVec2 l_start(origin.x - (DRAW_LEN * scale), origin.y + (DRAW_LEN * scale));
	ImVec2 l_end(origin.x + (DRAW_LEN * scale), origin.y - (DRAW_LEN * scale));

	canvas->AddLine(l_start, l_end, BLACK);
}