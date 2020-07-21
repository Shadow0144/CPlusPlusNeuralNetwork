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

float LeakyReLUFunction::getA() 
{ 
	return a;
}

void LeakyReLUFunction::setA(float a) 
{
	this->a = a;
}

int LeakyReLUFunction::numOutputs()
{
	return 1;
}

void LeakyReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, float scale)
{
	/*const Scalar BLACK(0, 0, 0);

	float slope = weights.getParameters().at<float>(0);
	float inv_slope = 1.0f / abs(slope);
	float x1, x2, y1, y2;
	if (slope > 0.0f)
	{
		x1 = -1.0f;
		x2 = +min(1.0f, inv_slope);
		y1 = -a;
		y2 = (x2 * slope);
	}
	else
	{
		x1 = -min(1.0f, inv_slope);
		x2 = 1.0f;
		y1 = (x1 * slope);
		y2 = a;
	}

	Point l_start(canvas.offset.x + ((int)(DRAW_LEN * x1)), canvas.offset.y - ((int)(DRAW_LEN * y1)));
	Point l_mid(canvas.offset.x, canvas.offset.y);
	Point l_end(canvas.offset.x + ((int)(DRAW_LEN * x2)), canvas.offset.y - ((int)(DRAW_LEN * y2)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_mid, BLACK, 1, LINE_8);
	line(canvas.canvas, l_mid, l_end, BLACK, 1, LINE_8);*/
}