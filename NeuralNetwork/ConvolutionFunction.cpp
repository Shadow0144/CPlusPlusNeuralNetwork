#include "ConvolutionFunction.h"

ConvolutionFunction::ConvolutionFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersOne(numInputs);
}

MatrixXd ConvolutionFunction::feedForward(MatrixXd input)
{
	return input;
}

MatrixXd ConvolutionFunction::backPropagate(MatrixXd lastInput, MatrixXd errors)
{
	weights.setDeltaParameters(-ALPHA * lastInput.transpose() * 0.0);
	return errors;
}

bool ConvolutionFunction::hasBias()
{
	return false;
}

int ConvolutionFunction::numOutputs()
{
	return numInputs;
}

void ConvolutionFunction::draw(NetworkVisualizer canvas)
{
	/*const Scalar BLACK(0, 0, 0);

	Point l_start(canvas.offset.x - DRAW_LEN, canvas.offset.y - ((int)(-DRAW_LEN)));
	Point l_end(canvas.offset.x + DRAW_LEN, canvas.offset.y - ((int)(DRAW_LEN)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);*/
}