#include "SoftmaxFunction.h"

#include <iostream>

SoftmaxFunction::SoftmaxFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersZero(0);
}

MatrixXd SoftmaxFunction::feedForward(MatrixXd inputs)
{
	return lastOutput;
}

MatrixXd SoftmaxFunction::backPropagate(MatrixXd lastInput, MatrixXd errors)
{
	double errorSum = errors.sum();
	double reLUPrime = (lastOutput(0) >= 0.0) ? 1.0 : 0.0;
	MatrixXd prime = MatrixXd::Ones(1, 1) * reLUPrime;
	MatrixXd sigma = errorSum * prime;

	weights.setDeltaParameters(-ALPHA * lastInput.transpose() * sigma);

	// Strip away the bias parameter and weight the sigma by the incoming weights
	MatrixXd weightsPrime = weights.getParameters().block(0, 0, (numInputs - 1), 1);

	return sigma * weightsPrime.transpose();
}

bool SoftmaxFunction::hasBias()
{
	return false;
}

int SoftmaxFunction::numOutputs()
{
	return numInputs;
}

void SoftmaxFunction::draw(ImDrawList* canvas, ImVec2 origin, float scale)
{
	/*const Scalar BLACK(0, 0, 0);

	float slope = weights.getParameters().at<float>(0);
	float inv_slope = 1.0f / abs(slope);
	float x1, x2, y1, y2;
	if (slope > 0.0f)
	{
		x1 = -1.0f;
		x2 = +min(1.0f, inv_slope);
		y1 = 0.0f;
		y2 = (x2 * slope);
	}
	else
	{
		x1 = -min(1.0f, inv_slope);
		x2 = 1.0f;
		y1 = (x1 * slope);
		y2 = 0.0f;
	}

	Point l_start(canvas.offset.x + ((int)(DRAW_LEN * x1)), canvas.offset.y - ((int)(DRAW_LEN * y1)));
	Point l_mid(canvas.offset.x, canvas.offset.y);
	Point l_end(canvas.offset.x + ((int)(DRAW_LEN * x2)), canvas.offset.y - ((int)(DRAW_LEN * y2)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_mid, BLACK, 1, LINE_8);
	line(canvas.canvas, l_mid, l_end, BLACK, 1, LINE_8);*/
}