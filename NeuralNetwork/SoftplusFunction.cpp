#include "SoftplusFunction.h"

#include <iostream>

SoftplusFunction::SoftplusFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->numOutputs = 1;
	this->weights.setParametersRandom(numInputs);
}

double softplus(double value, double k)
{
	return (log(1 + exp(k * value)) / k);
}

MatrixXd SoftplusFunction::feedForward(MatrixXd inputs)
{
	lastOutput = inputs * weights.getParameters();
	lastOutput(0) = softplus(lastOutput(0), k);
	return lastOutput;
}

MatrixXd SoftplusFunction::backPropagate(MatrixXd lastInput, MatrixXd errors)
{
	double errorSum = errors.sum();
	MatrixXd z = lastInput * weights.getParameters();
	double softPlusPrime = 1.0 / (1.0 + exp(-k * z(0)));
	MatrixXd prime = MatrixXd::Ones(1, 1) * softPlusPrime;
	MatrixXd sigma = errorSum * prime;

	weights.incrementDeltaParameters(-ALPHA * lastInput.transpose() * sigma);

	// Strip away the bias parameter and weight the sigma by the incoming weights
	MatrixXd weightsPrime = weights.getParameters().block(0, 0, (numInputs - 1), numOutputs);

	return weightsPrime * sigma.transpose();
}

bool SoftplusFunction::hasBias()
{
	return true;
}

double SoftplusFunction::getK() 
{
	return k;
}

void SoftplusFunction::setK(double k)
{
	this->k = k;
}

void SoftplusFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	int r = 3;
	double range = 3.0;

	int resolution = (r * 4) + 1;
	MatrixXd sP(resolution, 2);
	for (int r = 0; r < resolution; r++)
	{
		sP(r, 0) = range * (2.0 * r) / (resolution - 1.0) - range;
		sP(r, 1) = softplus(sP(r, 0), k);
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
	}
}