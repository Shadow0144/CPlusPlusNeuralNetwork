#include "IdentityFunction.h"

IdentityFunction::IdentityFunction(size_t numUnits, size_t incomingUnits)
{
	this->hasBias = false;
}

xt::xarray<double> IdentityFunction::feedForward(const xt::xarray<double>& input)
{
	return input;
}

xt::xarray<double> IdentityFunction::backPropagate(const xt::xarray<double>& sigmas)
{
	return sigmas;
}

void IdentityFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	ImVec2 l_start(origin.x - (DRAW_LEN * scale), origin.y + (DRAW_LEN * scale));
	ImVec2 l_end(origin.x + (DRAW_LEN * scale), origin.y - (DRAW_LEN * scale));

	canvas->AddLine(l_start, l_end, BLACK);
}