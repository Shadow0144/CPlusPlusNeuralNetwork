#include "DotProductFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;

DotProductFunction::DotProductFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

Mat DotProductFunction::feedForward(Mat inputs)
{
	Mat result = inputs * weights.getParameters();
	return result;
}

Mat DotProductFunction::backPropagate(Mat lastInput, Mat errors)
{
	// TODO: Make cleaner
	Scalar errorSum = cv::sum(errors);
	float errorSumF = ((float)(errorSum[0]));
	Mat prime = Mat::ones(1, 1, CV_32FC1);
	Mat sigma = errorSumF * prime;
	
	weights.setDeltaParameters(ALPHA * lastInput.t() * sigma);

	return sigma * weights.getParameters().t();
}

void DotProductFunction::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);

	Point l_start(canvas.offset.x - draw_len, canvas.offset.y - ((int)(-draw_len * weights.getParameters().at<float>(0))));
	Point l_end(canvas.offset.x + draw_len, canvas.offset.y - ((int)(draw_len * weights.getParameters().at<float>(0))));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, black, 1, LINE_8);
}