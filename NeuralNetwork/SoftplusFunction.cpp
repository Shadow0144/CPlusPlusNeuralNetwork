#include "SoftplusFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

SoftplusFunction::SoftplusFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

float softplus(float value, float k)
{
	return (log(1 + exp(k * value)) / k);
}

Mat SoftplusFunction::feedForward(Mat inputs)
{
	lastOutput = inputs * weights.getParameters();
	lastOutput.at<float>(0) = softplus(lastOutput.at<float>(0), k);
	return lastOutput;
}

Mat SoftplusFunction::backPropagate(Mat lastInput, Mat errors)
{
	// TODO: Make cleaner
	Scalar errorSum = cv::sum(errors);
	float errorSumF = ((float)(errorSum[0]));

	Mat z = lastInput * weights.getParameters();
	float softPlusPrime = 1.0f / (1.0f + exp(-k * z.at<float>(0)));
	Mat prime = Mat::ones(1, 1, CV_32FC1) * softPlusPrime;
	Mat sigma = errorSumF * prime;

	weights.setDeltaParameters(-ALPHA * lastInput.t() * sigma);

	// Strip away the bias parameter and weights the sigma by the incoming weights
	Mat weightsPrime = weights.getParameters();
	weightsPrime = weightsPrime(Rect(0, 0, 1, numInputs - 1)).t();

	return sigma * weightsPrime;
}

bool SoftplusFunction::hasBias()
{
	return true;
}

void SoftplusFunction::draw(DrawingCanvas canvas)
{
	const Scalar BLACK(0, 0, 0);
	const float STEP_SIZE = 0.1f;

	Function::draw(canvas);

	for (float i = -1.0f; i < 1.0f; i += STEP_SIZE)
	{
		int y1 = ((int)(DRAW_LEN * softplus(i * weights.getParameters().at<float>(0), k)));
		int y2 = ((int)(DRAW_LEN * softplus((i + STEP_SIZE) * weights.getParameters().at<float>(0), k)));
		Point l_start(canvas.offset.x + ((int)(DRAW_LEN * i)), canvas.offset.y - y1);
		Point l_end(canvas.offset.x + ((int)(DRAW_LEN * (i + STEP_SIZE))), canvas.offset.y - y2);
		line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);
	}
}