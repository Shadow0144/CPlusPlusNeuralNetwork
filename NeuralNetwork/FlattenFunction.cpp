#include "FlattenFunction.h"
#include "NeuralLayer.h"

#include "Test.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

FlattenFunction::FlattenFunction(int numOutputs)
{
	this->hasBias = false;
	this->numUnits = numOutputs; // Setting the number of units will set the correct value for number of outputs
}

xt::xarray<double> FlattenFunction::feedForward(const xt::xarray<double>& input)
{
	auto shape = input.shape();
	const int DIMS = shape.size();
	size_t newShape = 1;
	for (int i = 1; i < DIMS; i++)
	{
		newShape *= shape[i];
	}
	auto result = xt::xarray<double>(input);
	result.reshape({ shape[0], newShape });
	return result;
}

xt::xarray<double> FlattenFunction::backPropagate(const xt::xarray<double>& sigmas)
{
	auto sigmasPrime = xt::xarray<double>(sigmas);
	sigmasPrime.reshape(lastInput.shape());
	return sigmasPrime;
}

void FlattenFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);

	const double ARROW_WIDTH = 2.0 * scale;
	const double ARROW_HEIGHT = 4.0 * scale;

	const double RESCALE = DRAW_LEN * scale * RERESCALE;
	const double HALF_RESCALE = RESCALE / 2.0;
	const double QUAR_RESCALE = RESCALE / 4.0;

	const int NUM_BOXES = 4;

	// There is only one "neuron"
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(1, scale);
	origin.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, 0, scale);

	ImVec2 position(0, origin.y);

	// Draw arrow
	position.x = origin.x;
	canvas->AddLine(ImVec2(position.x + ARROW_WIDTH, position.y),
		ImVec2(position.x - ARROW_WIDTH, position.y - ARROW_HEIGHT), BLACK);
	canvas->AddLine(ImVec2(position.x + ARROW_WIDTH, position.y),
		ImVec2(position.x - ARROW_WIDTH, position.y + ARROW_HEIGHT), BLACK);

	//Draw left
	position.x = origin.x - (SHIFT * scale);
	ImVec2 start(position.x - RESCALE, position.y + RESCALE);
	ImVec2 end(position.x + RESCALE, position.y - RESCALE);

	canvas->AddRectFilled(start, end, WHITE);
	canvas->AddRect(start, end, BLACK);

	// Draw 4 boxes in a square
	ImVec2 zero_x_left(position.x - RESCALE, position.y);
	ImVec2 zero_x_right(position.x + RESCALE, position.y);
	canvas->AddLine(zero_x_left, zero_x_right, BLACK);
	ImVec2 zero_y_base(position.x, position.y + RESCALE);
	ImVec2 zero_y_top(position.x, position.y - RESCALE);
	canvas->AddLine(zero_y_base, zero_y_top, BLACK);

	// Draw right
	position.x = origin.x + (SHIFT * scale);
	start = ImVec2(position.x - RESCALE, position.y + QUAR_RESCALE);
	end = ImVec2(position.x + RESCALE, position.y - QUAR_RESCALE);

	canvas->AddRectFilled(start, end, WHITE);
	canvas->AddRect(start, end, BLACK);

	// Draw 4 boxes in a line
	ImVec2 leftTop(position.x - HALF_RESCALE, position.y + QUAR_RESCALE);
	ImVec2 leftBot(position.x - HALF_RESCALE, position.y - QUAR_RESCALE);
	ImVec2 centerTop(position.x, position.y + QUAR_RESCALE);
	ImVec2 centerBot(position.x, position.y - QUAR_RESCALE);
	ImVec2 rightTop(position.x + HALF_RESCALE, position.y + QUAR_RESCALE);
	ImVec2 rightBot(position.x + HALF_RESCALE, position.y - QUAR_RESCALE);
	canvas->AddLine(leftTop, leftBot, BLACK);
	canvas->AddLine(centerTop, centerBot, BLACK);
	canvas->AddLine(rightTop, rightBot, BLACK);
}