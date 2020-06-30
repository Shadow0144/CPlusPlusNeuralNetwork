#include "DotProductFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

DotProductFunction::DotProductFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

Mat DotProductFunction::feedForward(Mat inputs)
{
	Mat result(1, 1, CV_32FC1);
	float dot = ((float)(inputs.dot(weights.getParameters())));
	result.at<float>(0, 0) = dot;
	return result;
}

Mat DotProductFunction::backPropagate(Mat lastInput, Mat error)
{
	Mat sigma = cv::sum(error) * Mat::eye(1, 1, CV_32FC1);

	weights.setDeltaParameters(-ALPHA * sigma * lastInput);

	return sigma * weights.getParameters();
}

void DotProductFunction::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);

	Point l_start(canvas.offset.x - draw_len, canvas.offset.y - ((int)(-draw_len * weights.getParameters().at<float>(0))));
	Point l_end(canvas.offset.x + draw_len, canvas.offset.y - ((int)(draw_len * weights.getParameters().at<float>(0))));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, black, 1, LINE_8);
}