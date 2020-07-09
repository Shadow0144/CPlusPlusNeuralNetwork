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
	return inputs * weights.getParameters();
}

Mat DotProductFunction::backPropagate(Mat lastInput, Mat errors)
{
	// TODO: Make cleaner
	Scalar errorSum = cv::sum(errors);
	float errorSumF = ((float)(errorSum[0]));
	Mat prime = Mat::ones(1, 1, CV_32FC1);
	Mat sigma = errorSumF * prime;
	
	weights.setDeltaParameters(-ALPHA * lastInput.t() * sigma);

	// Strip away the bias parameter and weights the sigma by the incoming weights
	Mat weightsPrime = weights.getParameters();
	weightsPrime = weightsPrime(Rect(0, 0, 1, numInputs-1)).t();

	return sigma * weightsPrime;
}

bool DotProductFunction::hasBias()
{
	return true;
}

int DotProductFunction::numOutputs()
{
	return 1;
}

void DotProductFunction::draw(DrawingCanvas canvas)
{
	const Scalar BLACK(0, 0, 0);
	float slope = weights.getParameters().at<float>(0);
	float inv_slope = 1.0f / abs(slope);
	float x1 = -min(1.0f, inv_slope);
	float x2 = +min(1.0f, inv_slope);
	float y1 = x1 * slope;
	float y2 = x2 * slope;

	Point l_start(canvas.offset.x + ((int)(x1 * DRAW_LEN)), canvas.offset.y - ((int)(y1 * DRAW_LEN)));
	Point l_end(canvas.offset.x + ((int)(x2 * DRAW_LEN)), canvas.offset.y - ((int)(y2 * DRAW_LEN)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);
}