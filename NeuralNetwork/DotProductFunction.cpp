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

int DotProductFunction::numOutputs()
{
	return 1;
}

void DotProductFunction::draw(ImDrawList* canvas, ImVec2 origin, float scale)
{
	/*const Scalar BLACK(0, 0, 0);
	float slope = weights.getParameters().at<float>(0);
	float inv_slope = 1.0f / abs(slope);
	float x1 = -min(1.0f, inv_slope);
	float x2 = +min(1.0f, inv_slope);
	float y1 = x1 * slope;
	float y2 = x2 * slope;

	Point l_start(canvas.offset.x + ((int)(x1 * DRAW_LEN)), canvas.offset.y - ((int)(y1 * DRAW_LEN)));
	Point l_end(canvas.offset.x + ((int)(x2 * DRAW_LEN)), canvas.offset.y - ((int)(y2 * DRAW_LEN)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);*/
}