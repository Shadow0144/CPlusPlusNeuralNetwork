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