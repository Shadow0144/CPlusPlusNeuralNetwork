#include "FlattenFunction.h"

#include "Test.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

FlattenFunction::FlattenFunction()
{
	this->hasBias = false;
}

xt::xarray<double> FlattenFunction::feedForward(xt::xarray<double> input)
{
	lastInput = input;
	auto shape = lastInput.shape();
	const int DIMS = shape.size();
	size_t newShape = 1;
	for (int i = 1; i < DIMS; i++)
	{
		newShape *= shape[i];
	}
	auto result = input.reshape({ shape[0], newShape });
	/*print_dims(result);
	for (int i = 0; i < shape[0]; i++)
	{
		for (int j = 0; j < newShape; j++)
		{
			cout << result(i, j) << " ";
		}
	}
	cout << endl;*/
	return result;
}

xt::xarray<double> FlattenFunction::backPropagate(xt::xarray<double> sigmas)
{
	sigmas.reshape(lastInput.shape());
	return sigmas;
}

void FlattenFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	ImVec2 l_start(origin.x - (DRAW_LEN * scale), origin.y + (DRAW_LEN * scale));
	ImVec2 l_end(origin.x + (DRAW_LEN * scale), origin.y - (DRAW_LEN * scale));

	canvas->AddLine(l_start, l_end, BLACK);
}