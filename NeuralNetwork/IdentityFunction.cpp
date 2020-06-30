#include "IdentityFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

IdentityFunction::IdentityFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersOne(numInputs);
}

Mat IdentityFunction::feedForward(Mat input)
{
	return input;
}

Mat IdentityFunction::backPropagate(Mat lastInput, Mat errors)
{
	Mat sigma = cv::sum(errors) * Mat::zeros(1, 1, CV_32FC1);

	weights.setDeltaParameters(-ALPHA * sigma * lastInput);

	return sigma * weights.getParameters();
}

void IdentityFunction::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);

	Point l_start(canvas.offset.x - draw_len, canvas.offset.y - ((int)(-draw_len)));
	Point l_end(canvas.offset.x + draw_len, canvas.offset.y - ((int)(draw_len)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, black, 1, LINE_8);
}