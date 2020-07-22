#include "LeakyReLUFunction.h"

#include <iostream>

using namespace std;

LeakyReLUFunction::LeakyReLUFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

MatrixXd LeakyReLUFunction::feedForward(MatrixXd inputs)
{
	lastOutput = inputs * weights.getParameters();
	double lOut = lastOutput(0);
	lastOutput(0) = max(a * lOut, lOut);
	return lastOutput;
}

MatrixXd LeakyReLUFunction::backPropagate(MatrixXd lastInput, MatrixXd errors)
{
	double errorSum = errors.sum();
	double reLUPrime = (lastOutput(0) > 0.0) ? 1.0 : a;
	MatrixXd prime = MatrixXd::Ones(1, 1) * reLUPrime;
	MatrixXd sigma = errorSum * prime;

	weights.setDeltaParameters(-ALPHA * lastInput.transpose() * sigma);

	// Strip away the bias parameter and weight the sigma by the incoming weights
	MatrixXd weightsPrime = weights.getParameters().block(0, 0, (numInputs - 1), 1);

	return sigma * weightsPrime.transpose();
}

bool LeakyReLUFunction::hasBias()
{
	return true;
}

double LeakyReLUFunction::getA() 
{ 
	return a;
}

void LeakyReLUFunction::setA(double a) 
{
	this->a = a;
}

int LeakyReLUFunction::numOutputs()
{
	return 1;
}

void LeakyReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	double slope = weights.getParameters()(0);
	double inv_slope = 1.0 / abs(slope);
	double x1, x2, y1, y2;
	if (slope > 0.0f)
	{
		x1 = -1.0;
		x2 = +min(1.0, inv_slope);
		y1 = -a;
		y2 = (x2 * slope);
	}
	else
	{
		x1 = -min(1.0, inv_slope);
		x2 = 1.0;
		y1 = (x1 * slope);
		y2 = a;
	}

	ImVec2 l_start(origin.x + (DRAW_LEN * x1 * scale), origin.y - (DRAW_LEN * y1 * scale));
	ImVec2 l_mid(origin.x, origin.y);
	ImVec2 l_end(origin.x + (DRAW_LEN * x2 * scale), origin.y - (DRAW_LEN * y2 * scale));

	canvas->AddLine(l_start, l_mid, BLACK);
	canvas->AddLine(l_mid, l_end, BLACK);
}