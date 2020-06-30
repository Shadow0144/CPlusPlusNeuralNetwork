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
	Mat result(1, 1, CV_32FC1);
	float dot = ((float)(inputs.dot(weights.getParameters())));
	float reLUDot = (dot >= 0.0f) ? dot : 0.0f;
	result.at<float>(0, 0) = reLUDot;
	return result;
}

Mat ReLUFunction::backPropagate(Mat lastInput, Mat errors)
{
	// TODO: Make cleaner
	Scalar errorSum = cv::sum(errors);
	float errorSumF = ((float)(errorSum[0]));

	float dot = ((float)(lastInput.dot(weights.getParameters())));
	float reLUPrime = (dot >= 0.0f) ? 1.0f : 0.0f;
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

void ReLUFunction::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);

	int left = ((int)(-draw_len * weights.getParameters().at<float>(0)));
	left = (left >= 0) ? left : 0;
	int right = ((int)(draw_len * weights.getParameters().at<float>(0)));
	right = (right >= 0) ? right : 0;
	Point l_start(canvas.offset.x - draw_len, canvas.offset.y - left);
	Point l_mid(canvas.offset.x, canvas.offset.y);
	Point l_end(canvas.offset.x + draw_len, canvas.offset.y - right);

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_mid, black, 1, LINE_8);
	line(canvas.canvas, l_mid, l_end, black, 1, LINE_8);
}