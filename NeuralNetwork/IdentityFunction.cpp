#include "IdentityFunction.h"

IdentityFunction::IdentityFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->weights.setParametersOne(numInputs);
}

MatrixXd IdentityFunction::feedForward(MatrixXd input)
{
	return input;
}

MatrixXd IdentityFunction::backPropagate(MatrixXd lastInput, MatrixXd errors)
{
	weights.setDeltaParameters(-ALPHA * lastInput.transpose() * 0.0);
	return errors;
}

bool IdentityFunction::hasBias()
{
	return false;
}

int IdentityFunction::numOutputs()
{
	return numInputs;
}

void IdentityFunction::draw(ImDrawList* canvas, ImVec2 origin, float scale)
{
	/*const Scalar BLACK(0, 0, 0);

	Point l_start(canvas.offset.x - DRAW_LEN, canvas.offset.y - ((int)(-DRAW_LEN)));
	Point l_end(canvas.offset.x + DRAW_LEN, canvas.offset.y - ((int)(DRAW_LEN)));

	Function::draw(canvas);

	line(canvas.canvas, l_start, l_end, BLACK, 1, LINE_8);*/
}