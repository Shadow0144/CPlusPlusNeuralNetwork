#include "ActivationFunction/ActivationFunction.h"
#include "NeuralLayer/NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>
#pragma warning(pop)

#include "Test.h"

using namespace std;

xt::xarray<double> ActivationFunction::feedForwardTrain(const xt::xarray<double>& inputs)
{
	lastInput = inputs;
	lastOutput = feedForward(inputs);
	return lastOutput;
}

double ActivationFunction::activate(double z) const
{
	return z;
}

double ActivationFunction::applyBackPropagate()
{
	// Do nothing
	return 0;
}

double ActivationFunction::getRegularizationLoss(double lambda1, double lambda2) const
{
	return 0.0;
}

 double ActivationFunction::getParameter(const std::string& parameterName) const
 {
	 throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
 }

 void ActivationFunction::setParameter(const std::string& parameterName, double value)
 {
	 throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
 }

 void ActivationFunction::saveParameters(std::string fileName)
 {
	 // Do nothing
 }

 void ActivationFunction::loadParameters(std::string fileName)
 {
	 // Do nothing
 }

 void ActivationFunction::substituteParameters(Optimizer* optimizer)
 {
	 // Do nothing
 }

 void ActivationFunction::restoreParameters(Optimizer* optimizer)
 {
	 // Do nothing
 }

std::vector<size_t> ActivationFunction::getOutputShape(std::vector<size_t> outputShape) const
{
	return outputShape;
}

double ActivationFunction::sigmoid(double z) const
{
	return (1.0 / (1.0 + exp(-z)));
}

xt::xarray<double> ActivationFunction::sigmoid(const xt::xarray<double>& z) const
{
	return (1.0 / (1.0 + exp(-z)));
}

xt::xarray<double> ActivationFunction::approximateBezier(const xt::xarray<double>& points) const
{
	xt::xarray<double> T =
			{ { 1.0,       0.0,       0.0,			0.0 },
			  { 1.0,	1.0 / 3.0,	1.0 / 9.0,	1.0 / 27.0 },
			  { 1.0,	2.0 / 3.0,	4.0 / 9.0,	8.0 / 27.0 },
			  { 1.0,       1.0,       1.0,			1.0 } };

	xt::xarray<double> M =
			{ { 1.0,	0.0,	0.0,	0.0},
			  {-3.0,	3.0,	0.0,	0.0},
			  {3.0,		-6.0,	3.0,	0.0},
			  {-1.0,	3.0,	-3.0,	1.0} }; // 4

	/*xt::xarray<double> M =
			{ { 1.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0 },
			  { -6.0,   6.0,	0.0,	0.0,	0.0,	0.0,	0.0 },
			  { 15.0,	-30.0,  15.0,   0.0,	0.0,	0.0,	0.0 },
			  { -20.0,  60.0,	-60.0,	-20.0,  0.0,	0.0,	0.0 },
			  { 15.0,	-60.0,  90.0,	-60.0,  15.0,   0.0,	0.0 },
			  { -6.0,	30.0,	-60.0,	60.0,	-30.0,  6.0,	0.0 },
			  { 1.0,	-6.0,	15.0,	-20.0,  15.0,	-6.0,   1.0 } }; // 7 */

	auto Mi = xt::linalg::pinv(M);
	auto TT = xt::transpose(T);
	auto A = xt::linalg::dot(Mi, xt::linalg::dot(xt::linalg::pinv(xt::linalg::dot(TT, T)), TT));

	return (xt::linalg::dot(A, points));
}

// TODO: Cap vertical draw
void ActivationFunction::approximateFunction(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	const xt::xarray<double> drawWeights = weights.getParameters();

	const double RANGE = 3.0; // Controls the range of the plot to display (-RANGE, RANGE)

	const int R = 6; // Controls the number of points to estimate
	const int RESOLUTION = (R * 4) + 1; // Resolution must be 4r+1 points
	const float RESCALE = (1.0 / RANGE) * NeuralLayer::DRAW_LEN * scale;

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);

	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double x = 0;
		double y = 0;
		double lastX = -1.0;
		int pointCount = 0;
		int inflection = 0;
		xt::xtensor_fixed<double, xt::xshape<RESOLUTION, 2> > graphPoints;
		for (int r = 0; r < RESOLUTION; r++)
		{
			double x = RANGE * (2.0 * r) / (RESOLUTION - 1.0) - RANGE;
			double y = activate(x * drawWeights(0, i));
			if (y < -RANGE)
			{
				if (inflection == -1 || r == 0)
				{
					// Skip to next point
				}
				else // inflection was 0 or 1
				{
					// Estimate a point between this one and last
					graphPoints(pointCount, 0) = (lastX + x) / 2.0;
					graphPoints(pointCount, 1) = -RANGE;
					pointCount++;
				}
				inflection = -1;
			}
			else if (y > RANGE)
			{
				if (inflection == 1 || r == 0)
				{
					// Skip to next point
				}
				else // inflection was 0 or -1
				{
					// Estimate a point between this one and last
					graphPoints(pointCount, 0) = (lastX + x) / 2.0;
					graphPoints(pointCount, 1) = RANGE;
					pointCount++;
				}
				inflection = 1;
			}
			else
			{
				if (inflection == -1 || inflection == 1)
				{
					// Estimate a point between this one and last
					graphPoints(pointCount, 0) = (lastX + x) / 2.0;
					graphPoints(pointCount, 1) = inflection * RANGE;
					pointCount++;
				}
				else { }
				graphPoints(pointCount, 0) = x;
				graphPoints(pointCount, 1) = y;
				pointCount++;
				inflection = 0;
			}
			lastX = x;
		}

		// If there's a number of points not divisible by 3, but the last inflection is 0, add another point
		if ((((pointCount - 1) % 3) != 0) && (inflection == 0))
		{
			double x = RANGE;
			double y = activate(x * drawWeights(0, i));
			graphPoints(pointCount, 0) = x;
			graphPoints(pointCount, 1) = y;
			pointCount++;
		}
		else { }

		pointCount = pointCount - 2; // Do not go over
		for (int d = 0; d < pointCount; d += 3) // Use the last point as a start of the next line
		{
			auto points = approximateBezier(xt::view(graphPoints, xt::range(d, d + 4), xt::all())); // Grab 4 points
			canvas->AddBezierCurve(
				ImVec2(position.x + (points(0, 0) * RESCALE), position.y - (points(0, 1) * RESCALE)),
				ImVec2(position.x + (points(1, 0) * RESCALE), position.y - (points(1, 1) * RESCALE)),
				ImVec2(position.x + (points(2, 0) * RESCALE), position.y - (points(2, 1) * RESCALE)),
				ImVec2(position.x + (points(3, 0) * RESCALE), position.y - (points(3, 1) * RESCALE)),
				BLACK, 1);
		}
	}
}

void ActivationFunction::draw(ImDrawList * canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);

	const double RESCALE = NeuralLayer::DRAW_LEN * scale;

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		ImVec2 start(position.x - RESCALE, position.y + RESCALE);
		ImVec2 end(position.x + RESCALE, position.y - RESCALE);

		canvas->AddRectFilled(start, end, WHITE);
		canvas->AddRect(start, end, BLACK);

		if (drawAxes)
		{
			ImVec2 zero_x_left(position.x - RESCALE, position.y);
			ImVec2 zero_x_right(position.x + RESCALE, position.y);
			canvas->AddLine(zero_x_left, zero_x_right, LIGHT_GRAY);
			ImVec2 zero_y_base(position.x, position.y + RESCALE);
			ImVec2 zero_y_top(position.x, position.y - RESCALE);
			canvas->AddLine(zero_y_base, zero_y_top, LIGHT_GRAY);
		}
		else { }
	}
}

void ActivationFunction::drawConversion(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);

	const double ARROW_WIDTH = 2.0 * scale;
	const double ARROW_HEIGHT = 4.0 * scale;

	const double RESCALE = NeuralLayer::DRAW_LEN * scale * NeuralLayer::RERESCALE;

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		// Draw arrow
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddLine(ImVec2(position.x + ARROW_WIDTH, position.y),
			ImVec2(position.x - ARROW_WIDTH, position.y - ARROW_HEIGHT), BLACK);
		canvas->AddLine(ImVec2(position.x + ARROW_WIDTH, position.y),
			ImVec2(position.x - ARROW_WIDTH, position.y + ARROW_HEIGHT), BLACK);

		//Draw left
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale) - (NeuralLayer::SHIFT * scale);
		ImVec2 start(position.x - RESCALE, position.y + RESCALE);
		ImVec2 end(position.x + RESCALE, position.y - RESCALE);

		canvas->AddRectFilled(start, end, WHITE);
		canvas->AddRect(start, end, BLACK);

		if (drawAxes)
		{
			ImVec2 zero_x_left(position.x - RESCALE, position.y);
			ImVec2 zero_x_right(position.x + RESCALE, position.y);
			canvas->AddLine(zero_x_left, zero_x_right, LIGHT_GRAY);
			ImVec2 zero_y_base(position.x, position.y + RESCALE);
			ImVec2 zero_y_top(position.x, position.y - RESCALE);
			canvas->AddLine(zero_y_base, zero_y_top, LIGHT_GRAY);
		}
		else { }

		// Draw right
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale) + (NeuralLayer::SHIFT * scale);
		start = ImVec2(position.x - RESCALE, position.y + RESCALE);
		end = ImVec2(position.x + RESCALE, position.y - RESCALE);

		canvas->AddRectFilled(start, end, WHITE);
		canvas->AddRect(start, end, BLACK);

		if (drawAxes)
		{
			ImVec2 zero_x_left(position.x - RESCALE, position.y);
			ImVec2 zero_x_right(position.x + RESCALE, position.y);
			canvas->AddLine(zero_x_left, zero_x_right, LIGHT_GRAY);
			ImVec2 zero_y_base(position.x, position.y + RESCALE);
			ImVec2 zero_y_top(position.x, position.y - RESCALE);
			canvas->AddLine(zero_y_base, zero_y_top, LIGHT_GRAY);
		}
		else { }
	}
}