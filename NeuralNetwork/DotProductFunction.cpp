#include "DotProductFunction.h"

#include <iostream>

using namespace std;

DotProductFunction::DotProductFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

MatrixXd DotProductFunction::feedForward(MatrixXd inputs)
{
	return inputs * weights.getParameters();
}

MatrixXd DotProductFunction::backPropagate(MatrixXd lastInput, MatrixXd errors)
{
	double errorSum = errors.sum();
	MatrixXd prime = MatrixXd::Ones(1, 1);
	MatrixXd sigma = errorSum * prime;
	
	weights.setDeltaParameters(-ALPHA * lastInput.transpose() * sigma);

	// Strip away the bias parameter and weight the sigma by the incoming weights
	MatrixXd weightsPrime = weights.getParameters().block(0, 0, (numInputs - 1), 1);

	return sigma * weightsPrime.transpose();
}

bool DotProductFunction::hasBias()
{
	return true;
}

int DotProductFunction::getNumOutputs()
{
	return 1;
}

void DotProductFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	double slope = weights.getParameters()(0);
	double inv_slope = 1.0 / abs(slope);
	double x1 = -min(1.0, inv_slope);
	double x2 = +min(1.0, inv_slope);
	double y1 = x1 * slope;
	double y2 = x2 * slope;

	ImVec2 l_start(origin.x + (x1 * DRAW_LEN * scale), origin.y - (y1 * DRAW_LEN * scale));
	ImVec2 l_end(origin.x + (x2 * DRAW_LEN * scale), origin.y - (y2 * DRAW_LEN * scale));

	canvas->AddLine(l_start, l_end, BLACK);
}