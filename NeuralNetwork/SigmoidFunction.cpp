#include "SigmoidFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

SigmoidFunction::SigmoidFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

Mat SigmoidFunction::feedForward(Mat inputs)
{
	lastOutput = inputs * weights.getParameters();
	exp(-lastOutput, lastOutput);
	lastOutput = 1.0f / (1.0f + lastOutput);
	return lastOutput;
}

Mat SigmoidFunction::backPropagate(Mat lastInput, Mat errors)
{
	// TODO: Make cleaner
	Scalar errorSum = cv::sum(errors);
	float errorSumF = ((float)(errorSum[0]));

	Mat prime = lastOutput * (1 - lastOutput);
	Mat sigma = errorSumF * prime;

	weights.setDeltaParameters(-ALPHA * lastInput.t() * sigma);

	// Strip away the bias parameter and weights the sigma by the incoming weights
	Mat weightsPrime = weights.getParameters();
	weightsPrime = weightsPrime(Rect(0, 0, 1, numInputs-1)).t();

	return sigma * weightsPrime;
}

bool SigmoidFunction::hasBias()
{
	return true;
}

float sigmoid(float value)
{
	return (1.0f / (1.0f + exp(-value)));
}

void SigmoidFunction::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);
	const float step_size = 0.1f;

	Function::draw(canvas);

	for (float i = -1.0f; i < 1.0f; i += step_size)
	{
		int y1 = ((int)(draw_len * sigmoid(i * weights.getParameters().at<float>(0))));
		int y2 = ((int)(draw_len * sigmoid((i + step_size) * weights.getParameters().at<float>(0))));
		Point l_start(canvas.offset.x + ((int)(draw_len * i)), canvas.offset.y - y1);
		Point l_end(canvas.offset.x + ((int)(draw_len * (i + step_size))), canvas.offset.y - y2);
		line(canvas.canvas, l_start, l_end, black, 1, LINE_8);
	}
}