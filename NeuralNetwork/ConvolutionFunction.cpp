#include "ConvolutionFunction.h"

ConvolutionFunction::ConvolutionFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->numOutputs = 1;
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

void ConvolutionFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	/*const Scalar BLACK(0, 0, 0);

	Point l_start(canvas.offset.x - DRAW_LEN, canvas.offset.y - ((int)(-DRAW_LEN)));
	Point l_end(canvas.offset.x + DRAW_LEN, canvas.offset.y - ((int)(DRAW_LEN)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);*/
}