#include "HardSigmoidFunction.h"
#include "NeuralLayer.h"

using namespace std;

HardSigmoidFunction::HardSigmoidFunction(size_t incomingUnits, size_t numUnits)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// input x output -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
}

xt::xarray<double> HardSigmoidFunction::hard_sigmoid(xt::xarray<double> z)
{
	auto zero = (z < -2.5); // 0
	auto one = (z > 2.5); // 1
	auto mid = (1.0 - zero) * (1.0 - one);
	auto r = (0.2 * (z * mid) + 0.5 * mid) + one;
	return r;
}

xt::xarray<double> HardSigmoidFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	lastOutput = hard_sigmoid(dotProductResult);
	return lastOutput;
}

xt::xarray<double> HardSigmoidFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> HardSigmoidFunction::activationDerivative()
{
	return ((lastOutput < 1) * (lastOutput > 0)); // The slope is 1 between these two ranges, 0 otherwise
}

void HardSigmoidFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	const double RANGE = 6.0; // Controls the range of the plot to display (-RANGE, RANGE)
	float rescale = (1.0 / RANGE) * DRAW_LEN * scale;

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = weights.getParameters()(0, i);
		double inv_slope = 1.0 / abs(slope);
		double x1, x2, y1, y2;
		x1 = max(-2.5 * inv_slope, -RANGE);
		x2 = min(+2.5 * inv_slope, RANGE);
		if (slope > 0.0)
		{
			y1 = (x1 > -RANGE) ? -1.0 : +1.0;
		}
		else
		{
			y1 = (x1 > -RANGE) ? +1.0 : -1.0;
		}
		y2 = -y1;

		ImVec2 l_start(position.x + (-RANGE * rescale), position.y - (y1 * rescale));
		ImVec2 l_mid_left(position.x + (x1 * rescale), position.y - (y1 * rescale));
		ImVec2 l_mid_right(position.x + (x2 * rescale), position.y - (y2 * rescale));
		ImVec2 l_end(position.x + (RANGE * rescale), position.y - (y2 * rescale));

		if (x1 > -RANGE && x2 < +RANGE)
		{
			canvas->AddLine(l_start, l_mid_left, BLACK);
			canvas->AddLine(l_mid_left, l_mid_right, BLACK);
			canvas->AddLine(l_mid_right, l_end, BLACK);
		}
		else
		{
			canvas->AddLine(l_mid_left, l_mid_right, BLACK);
		}
	}
}