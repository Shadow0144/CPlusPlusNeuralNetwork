#include "ActivationFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

#include "Test.h"

using namespace std;
using namespace Eigen;

xt::xarray<double> ActivationFunction::feedForwardTrain(const xt::xarray<double>& inputs)
{
	lastInput = inputs;
	lastOutput = feedForward(inputs);
	return lastOutput;
}

double ActivationFunction::activate(double z)
{
	return z;
}
 void ActivationFunction::applyBackPropagate()
{
	// Do nothing
}

std::vector<size_t> ActivationFunction::getOutputShape(std::vector<size_t> outputShape)
{
	return outputShape;
}

MatrixXd ActivationFunction::approximateBezier(const MatrixXd& points)
{
	//int pCount = points.rows();
	//int k = pCount - 1;

	/*MatrixXd T(pCount, pCount);
	for (int i = 0; i < pCount; i++)
	{
		for (int j = 0; j < pCount; j++)
		{
			double jk = ((double)j) / ((double)k);
			T(k-i, k-j) = pow(jk, i);
		}
	}*/

	MatrixXd T(4, 4);
	T << 1.0,       0.0,       0.0,        0.0,
		 1.0, 1.0 / 3.0, 1.0 / 9.0, 1.0 / 27.0,
		 1.0, 2.0 / 3.0, 4.0 / 9.0, 8.0 / 27.0,
		 1.0,       1.0,       1.0,        1.0;

	MatrixXd M(4, 4);
	M << 1,  0,  0,  0,
		-3,  3,  0,  0,
		 3, -6,  3,  0,
		-1,  3, -3,  1;

	/*MatrixXd M(7, 7);
	M <<     1,   0,   0,   0,   0,   0,   0,
			 -6,   6,   0,   0,   0,   0,   0,
			 15, -30,  15,   0,   0,   0,   0,
			-20,  60, -60, -20,   0,   0,   0,
			 15, -60,  90, -60,  15,   0,   0,
			 -6,  30, -60,  60, -30,   6,   0,
			  1,  -6,  15, -20,  15,  -6,   1;*/

	MatrixXd Mi = M.inverse();
	MatrixXd TT = T.transpose();
	MatrixXd A = Mi * (TT * T).inverse() * TT;

	return A * points;
}

// TODO: Cap vertical draw
void ActivationFunction::approximateFunction(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
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

		double x, y;
		double lastX = -1.0;
		int pointCount = 0;
		int inflection = 0;
		MatrixXd graphPoints(RESOLUTION, 2);
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
			MatrixXd points = approximateBezier(graphPoints.block(d, 0, 4, 2)); // Grab 4 points
			canvas->AddBezierCurve(
				ImVec2(position.x + (points(0, 0) * RESCALE), position.y - (points(0, 1) * RESCALE)),
				ImVec2(position.x + (points(1, 0) * RESCALE), position.y - (points(1, 1) * RESCALE)),
				ImVec2(position.x + (points(2, 0) * RESCALE), position.y - (points(2, 1) * RESCALE)),
				ImVec2(position.x + (points(3, 0) * RESCALE), position.y - (points(3, 1) * RESCALE)),
				BLACK, 1);
		}
	}
}

void ActivationFunction::draw(ImDrawList * canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
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

void ActivationFunction::drawConversion(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
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