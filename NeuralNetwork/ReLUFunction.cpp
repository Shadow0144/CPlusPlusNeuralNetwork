#include "ReLUFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

ReLUFunction::ReLUFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

Mat ReLUFunction::feedForward(Mat inputs)
{
	lastOutput = inputs * weights.getParameters();
	lastOutput.at<float>(0) = max(0.0f, lastOutput.at<float>(0));
	return lastOutput;
}

Mat ReLUFunction::backPropagate(Mat lastInput, Mat errors)
{
	// TODO: Make cleaner
	Scalar errorSum = cv::sum(errors);
	float errorSumF = ((float)(errorSum[0]));

	float reLUPrime = (lastOutput.at<float>(0) > 0.0f) ? 1.0f : 0.0f;
	Mat prime = Mat::ones(1, 1, CV_32FC1) * reLUPrime;
	Mat sigma = errorSumF * prime;

	weights.setDeltaParameters(-ALPHA * lastInput.t() * sigma);

	// Strip away the bias parameter and weights the sigma by the incoming weights
	Mat weightsPrime = weights.getParameters();
	weightsPrime = weightsPrime(Rect(0, 0, 1, numInputs-1)).t();

	return sigma * weightsPrime;
}

bool ReLUFunction::hasBias()
{
	return true;
}

int ReLUFunction::numOutputs()
{
	return 1;
}

void ReLUFunction::draw(DrawingCanvas canvas)
{
	const Scalar BLACK(0, 0, 0);

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
	line(canvas.canvas, l_mid, l_end, BLACK, 1, LINE_8);
}