#include "SigmoidFunction.h"

SigmoidFunction::SigmoidFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

double sigmoid(double value)
{
	return (1.0f / (1.0f + exp(-value)));
}

MatrixXd SigmoidFunction::feedForward(MatrixXd inputs)
{
	lastOutput = inputs * weights.getParameters();
	lastOutput(0) = sigmoid(lastOutput(0));
	return lastOutput;
}

MatrixXd SigmoidFunction::backPropagate(MatrixXd lastInput, MatrixXd errors)
{
	double errorSum = errors.sum();
	MatrixXd prime = lastOutput * (MatrixXd::Ones(lastOutput.rows(), lastOutput.cols()) - lastOutput);
	MatrixXd sigma = errorSum * prime;

	weights.setDeltaParameters(-ALPHA * lastInput.transpose() * sigma);

	// Strip away the bias parameter and weight the sigma by the incoming weights
	MatrixXd weightsPrime = weights.getParameters().block(0, 0, (numInputs - 1), 1);

	return sigma * weightsPrime.transpose();
}

bool SigmoidFunction::hasBias()
{
	return true;
}

int SigmoidFunction::numOutputs()
{
	return 1;
}

void SigmoidFunction::draw(NetworkVisualizer canvas)
{
	/*const Scalar BLACK(0, 0, 0);
	const float STEP_SIZE = 0.1f;

	Function::draw(canvas);

	for (float i = -1.0f; i < 1.0f; i += STEP_SIZE)
	{
		int y1 = ((int)(DRAW_LEN * sigmoid(i * weights.getParameters().at<float>(0))));
		int y2 = ((int)(DRAW_LEN * sigmoid((i + STEP_SIZE) * weights.getParameters().at<float>(0))));
		Point l_start(canvas.offset.x + ((int)(DRAW_LEN * i)), canvas.offset.y - y1);
		Point l_end(canvas.offset.x + ((int)(DRAW_LEN * (i + STEP_SIZE))), canvas.offset.y - y2);
		line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);
	}*/
}