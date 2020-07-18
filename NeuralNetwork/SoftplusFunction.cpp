#include "SoftplusFunction.h"

#include <iostream>

SoftplusFunction::SoftplusFunction(int numInputs)
{
	this->numInputs = numInputs;
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

	weights.setDeltaParameters(-ALPHA * lastInput.transpose() * sigma);

	// Strip away the bias parameter and weight the sigma by the incoming weights
	MatrixXd weightsPrime = weights.getParameters().block(0, 0, (numInputs - 1), 1);

	return sigma * weightsPrime.transpose();
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

int SoftplusFunction::numOutputs()
{
	return 1;
}

void SoftplusFunction::draw(NetworkVisualizer canvas)
{
	/*const Scalar BLACK(0, 0, 0);
	const float STEP_SIZE = 0.1f;

	Function::draw(canvas);

	for (float i = -1.0f; i < 1.0f; i += STEP_SIZE)
	{
		int y1 = ((int)(DRAW_LEN * softplus(i * weights.getParameters().at<float>(0), k)));
		int y2 = ((int)(DRAW_LEN * softplus((i + STEP_SIZE) * weights.getParameters().at<float>(0), k)));
		Point l_start(canvas.offset.x + ((int)(DRAW_LEN * i)), canvas.offset.y - y1);
		Point l_end(canvas.offset.x + ((int)(DRAW_LEN * (i + STEP_SIZE))), canvas.offset.y - y2);
		line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);
	}*/
}