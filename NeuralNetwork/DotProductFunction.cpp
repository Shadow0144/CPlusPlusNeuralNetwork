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

void DotProductFunction::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);

	Point l_start(canvas.offset.x - draw_len, canvas.offset.y - ((int)(-draw_len * weights.getParameters().at<float>(0))));
	Point l_end(canvas.offset.x + draw_len, canvas.offset.y - ((int)(draw_len * weights.getParameters().at<float>(0))));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, black, 1, LINE_8);
}