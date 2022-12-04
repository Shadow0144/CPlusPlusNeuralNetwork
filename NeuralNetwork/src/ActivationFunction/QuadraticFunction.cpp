#include "ActivationFunction/QuadraticFunction.h"
#include "NeuralLayer/NeuralLayer.h"

using namespace std;

QuadraticFunction::QuadraticFunction()
{

}

xt::xarray<double> QuadraticFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return pow(inputs, 2.0);
}

xt::xarray<double> QuadraticFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return (sigmas * (2.0 * lastInput));
}

void QuadraticFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	xt::xarray<double> drawWeights = weights.getParameters();

	const double RANGE = 3.0; // Controls the range of the plot to display (-RANGE, RANGE)
	const double RANGE_SQRT = sqrt(RANGE);
	float rescale = (1.0 / RANGE) * NeuralLayer::DRAW_LEN * scale;

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = drawWeights(0, i);
		double sq_inv_slope = pow(1.0 / slope, 2.0);
		double x, y;
		if (sq_inv_slope < RANGE)
		{
			y = RANGE;
			x = RANGE_SQRT / abs(slope);
		}
		else
		{
			x = RANGE;
			y = pow(RANGE * slope, 2.0);
		}

		double x1_3 = (x / 3.0);
		double y1_3 = pow(x1_3 * slope, 2.0);
		double x2_3 = 2.0 * x1_3;
		double y2_3 = pow(x2_3 * slope, 2.0);

		ImVec2 l_start(position.x - (x * rescale), position.y - (y * rescale));
		ImVec2 l_start_left(position.x - (x2_3 * rescale), position.y - (y2_3 * rescale));
		ImVec2 l_start_right(position.x - (x1_3 * rescale), position.y - (y1_3 * rescale));

		ImVec2 l_mid(position.x, position.y);

		ImVec2 l_end_left(position.x + (x1_3 * rescale), position.y - (y1_3 * rescale));
		ImVec2 l_end_right(position.x + (x2_3 * rescale), position.y - (y2_3 * rescale));
		ImVec2 l_end(position.x + (x * rescale), position.y - (y * rescale));

		canvas->AddBezierCurve(l_start, l_start_left, l_start_right, l_mid, BLACK, 1);
		canvas->AddBezierCurve(l_mid, l_end_left, l_end_right, l_end, BLACK, 1);
	}
}