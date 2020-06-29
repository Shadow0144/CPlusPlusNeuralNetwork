#include "SigmoidFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

SigmoidFunction::SigmoidFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParameters(numInputs);
}

Mat SigmoidFunction::feedForward(Mat inputs)
{
	Mat result(1, 1, CV_32FC1);
	float dot = ((float)(inputs.dot(weights.getParameters())));
	float sigmoidDot = 1.0f / (1.0f + exp(-dot));
	result.at<float>(0, 0) = sigmoidDot;
	return result;
}

Mat SigmoidFunction::backPropagate(Mat error)
{
	Mat mat;
	return mat;
}

void SigmoidFunction::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);

	Point l_start(canvas.offset.x - draw_len, canvas.offset.y - ((int)(-draw_len * weights.getParameters().at<float>(0))));
	Point l_end(canvas.offset.x + draw_len, canvas.offset.y - ((int)(draw_len * weights.getParameters().at<float>(0))));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, black, 1, LINE_8);
}