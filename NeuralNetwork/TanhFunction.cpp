#include "TanhFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

TanhFunction::TanhFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParameters(numInputs);
}

Mat TanhFunction::feedForward(Mat inputs)
{
	Mat result(1, 1, CV_32FC1);
	float dot = ((float)(inputs.dot(weights.getParameters())));
	float tanhDot = tanh(dot);
	result.at<float>(0, 0) = tanhDot;
	return result;
}

Mat TanhFunction::backPropagate(Mat error)
{
	Mat mat;
	return mat;
}

void TanhFunction::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);

	Point l_start(canvas.offset.x - draw_len, canvas.offset.y - ((int)(-draw_len * weights.getParameters().at<float>(0))));
	Point l_end(canvas.offset.x + draw_len, canvas.offset.y - ((int)(draw_len * weights.getParameters().at<float>(0))));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, black, 1, LINE_8);
}