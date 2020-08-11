#include "SigmoidFunction.h"

SigmoidFunction::SigmoidFunction(size_t incomingUnits, size_t numUnits)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// incoming x current -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
}
	
xt::xarray<double> SigmoidFunction::sigmoid(xt::xarray<double> z)
{
	return (1.0 / (1.0 + exp(-z)));
}

xt::xarray<double> SigmoidFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	return sigmoid(dotProductResult);
}

xt::xarray<double> SigmoidFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> SigmoidFunction::activationDerivative()
{
	return lastOutput * (1.0 - lastOutput);
}

void SigmoidFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	/*Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	int r = 3;
	double range = 3.0;

	int resolution = (r * 4) + 1;
	MatrixXd sP(resolution, 2);
	for (int r = 0; r < resolution; r++)
	{
		sP(r, 0) = range * (2.0 * r) / (resolution - 1.0) - range;
		sP(r, 1) = sigmoid(sP(r, 0));
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