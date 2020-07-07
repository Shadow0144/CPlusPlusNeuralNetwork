#include "TanhFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

TanhFunction::TanhFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersRandom(numInputs);
}

Mat TanhFunction::feedForward(Mat inputs)
{
	lastOutput = inputs * weights.getParameters();
	lastOutput.at<float>(0) = tanh(lastOutput.at<float>(0));
	return lastOutput;
}

Mat TanhFunction::backPropagate(Mat lastInput, Mat errors)
{
	// TODO: Make cleaner
	Scalar errorSum = cv::sum(errors);
	float errorSumF = ((float)(errorSum[0]));

	Mat prime = 1 - (lastOutput * lastOutput);
	Mat sigma = errorSumF * prime;

	weights.setDeltaParameters(-ALPHA * lastInput.t() * sigma);

	// Strip away the bias parameter and weights the sigma by the incoming weights
	Mat weightsPrime = weights.getParameters();
	weightsPrime = weightsPrime(Rect(0, 0, 1, numInputs-1)).t();

	return sigma * weightsPrime;
}

bool TanhFunction::hasBias()
{
	return true;
}

void TanhFunction::draw(DrawingCanvas canvas)
{
	const Scalar BLACK(0, 0, 0);
	const float STEP_SIZE = 0.1f;

	Function::draw(canvas);

	for (float i = -1.0f; i < 1.0f; i += STEP_SIZE)
	{
		int y1 = ((int)(DRAW_LEN * tanh(i * weights.getParameters().at<float>(0))));
		int y2 = ((int)(DRAW_LEN * tanh((i + STEP_SIZE) * weights.getParameters().at<float>(0))));
		Point l_start(canvas.offset.x + ((int)(DRAW_LEN * i)), canvas.offset.y - y1);
		Point l_end(canvas.offset.x + ((int)(DRAW_LEN * (i + STEP_SIZE))), canvas.offset.y - y2);
		line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);
	}
}