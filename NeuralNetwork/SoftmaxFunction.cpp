#include "SoftmaxFunction.h"

#include <iostream>

using namespace std;

SoftmaxFunction::SoftmaxFunction(int numInputs, int numOutputs)
{
	this->numInputs = numInputs;
	this->numOutputs = numOutputs;
	this->weights.setParametersRandom(numInputs, numOutputs);
}

MatrixXd SoftmaxFunction::feedForward(MatrixXd inputs)
{
	double total = 0.0;

	lastOutput = MatrixXd(1, numOutputs);
	double c = -inputs.maxCoeff();
	for (int i = 0; i < numOutputs; i++)
	{
		MatrixXd z = inputs * weights.getParameters().col(i);
		lastOutput(i) = exp(z(0) + c);
		total += lastOutput(i);
	}
	lastOutput /= total;

	return lastOutput;
}

MatrixXd SoftmaxFunction::backPropagate(MatrixXd lastInput, MatrixXd errors)
{
	MatrixXd outputDiagonal = lastOutput.row(0).asDiagonal();
	MatrixXd lastOutputRep = lastOutput.replicate(numOutputs, 1);
	MatrixXd prime = -lastOutputRep.cwiseProduct(MatrixXd::Identity(numOutputs, numOutputs) - lastOutputRep.transpose());
	MatrixXd sigma = errors * prime;

	weights.setDeltaParameters(-ALPHA * lastInput.transpose() * sigma);

	// Strip away the bias parameter and weight the sigma by the incoming weights
	MatrixXd weightsPrime = weights.getParameters().block(0, 0, (numInputs - 1), numOutputs);

	return weightsPrime * sigma.transpose();
}

bool SoftmaxFunction::hasBias()
{
	return true;
}

void SoftmaxFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

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