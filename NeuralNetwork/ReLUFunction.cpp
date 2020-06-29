#include "ReLUFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

ReLUFunction::ReLUFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParameters(numInputs);
}

Mat ReLUFunction::feedForward(Mat inputs)
{
	Mat result(1, 1, CV_32FC1);
	float dot = ((float)(inputs.dot(weights.getParameters())));
	result.at<float>(0, 0) = dot;
	return result;
}

Mat ReLUFunction::backPropagate(Mat error)
{
	Mat mat;
	return mat;
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