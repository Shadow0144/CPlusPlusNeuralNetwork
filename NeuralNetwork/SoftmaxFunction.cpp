#include "SoftmaxFunction.h"

#include <iostream>
#include <math.h>

using namespace std;

SoftmaxFunction::SoftmaxFunction(size_t incomingUnits, int axis)
{
	this->hasBias = false;
	this->numInputs = incomingUnits;
	this->numOutputs = numInputs;
	this->axis = axis;
}

xt::xarray<double> SoftmaxFunction::feedForward(xt::xarray<double> inputs)
{
	int sumAxis = (axis > 0) ? (axis) : (inputs.dimension() + axis);

	double c = 0;// -0.1; // negative max per axis // TODO
	auto z = xt::exp(inputs + c);

	// We lose a dimension when summing, so broadcasting won't work without this trick
	auto shape = z.shape();
	shape[sumAxis] = 1;
	xt::xstrided_slice_vector dimensionView;
	for (int i = 0; i < sumAxis; i++)
	{
		dimensionView.push_back(xt::all());
	}
	dimensionView.push_back(0);
	dimensionView.push_back(xt::ellipsis());
	xt::xarray<double> total(shape);
	xt::strided_view(total, dimensionView) = xt::sum<double>(z, { sumAxis });
	
	lastOutput = z / total;

	return lastOutput;
}

xt::xarray<double> SoftmaxFunction::backPropagate(xt::xarray<double> sigmas)
{
	//auto shape = lastOutput.shape();
	//auto broadcasted = sigmas * xt::ones<double>(shape);
	auto newSigmas = sigmas;
	return newSigmas;
}

void SoftmaxFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	int r = 3;
	double range = 3.0;

	int resolution = (r * 4) + 1;
	/*MatrixXd simInput = MatrixXd::Zero(1, numInputs);
	simInput(0, numInputs - 1) = 1; // Bias
	MatrixXd sP(resolution, 2);
	for (int r = 0; r < resolution; r++)
	{
		// TODO: Clip if exceeds range
		sP(r, 0) = range * (2.0 * r) / (resolution - 1.0) - range;
		simInput(0, 0) = sP(r, 0);
		sP(r, 1) = feedForward(simInput)(0);
	}

	float rescale = (1.0 / range) * DRAW_LEN * scale;
	for (int d = 0; d < (resolution - 3); d += 3)
	{
		MatrixXd points = approximateBezier(sP.block(d, 0, 4, 2));
		canvas->AddBezierCurve(
			ImVec2(origin.x + (points(0, 0) * rescale), origin.y - (points(0, 1) * rescale)),
			ImVec2(origin.x + (points(1, 0) * rescale), origin.y - (points(1, 1) * rescale)),
			ImVec2(origin.x + (points(2, 0) * rescale), origin.y - (points(2, 1) * rescale)),
			ImVec2(origin.x + (points(3, 0) * rescale), origin.y - (points(3, 1) * rescale)),
			BLACK, 1);
	}*/
}