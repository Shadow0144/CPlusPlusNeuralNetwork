#include "ReLUFunction.h"

#include <iostream>

using namespace std;

ReLUFunction::ReLUFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

MatrixXd ReLUFunction::feedForward(MatrixXd inputs)
{
	lastOutput = inputs * weights.getParameters();
	lastOutput(0) = max(0.0, lastOutput(0));
	return lastOutput;
}

MatrixXd ReLUFunction::backPropagate(MatrixXd lastInput, MatrixXd errors)
{
	double errorSum = errors.sum();
	double reLUPrime = (lastOutput(0) > 0.0) ? 1.0 : 0.0;
	MatrixXd prime = MatrixXd::Ones(1, 1) * reLUPrime;
	MatrixXd sigma = errorSum * prime;

	weights.setDeltaParameters(-ALPHA * lastInput.transpose() * sigma);

	// Strip away the bias parameter and weight the sigma by the incoming weights
	MatrixXd weightsPrime = weights.getParameters().block(0, 0, (numInputs - 1), 1);

	return sigma * weightsPrime.transpose();
}

bool ReLUFunction::hasBias()
{
	return true;
}

int ReLUFunction::numOutputs()
{
	return 1;
}

void ReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	double slope = weights.getParameters()(0);
	double inv_slope = 1.0 / abs(slope);
	double x1, x2, y1, y2;
	if (slope > 0.0)
	{
		x1 = -1.0;
		x2 = +min(1.0, inv_slope);
		y1 = 0.0;
		y2 = (x2 * slope);
	}
	else
	{
		x1 = -min(1.0, inv_slope);
		x2 = 1.0;
		y1 = (x1 * slope);
		y2 = 0.0;
	}

	ImVec2 l_start(origin.x + (DRAW_LEN * x1 * scale), origin.y - (DRAW_LEN * y1 * scale));
	ImVec2 l_mid(origin.x, origin.y);
	ImVec2 l_end(origin.x + (DRAW_LEN * x2 * scale), origin.y - (DRAW_LEN * y2 * scale));

	canvas->AddLine(l_start, l_mid, BLACK);
	canvas->AddLine(l_mid, l_end, BLACK);
}