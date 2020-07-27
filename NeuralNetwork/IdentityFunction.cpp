#include "IdentityFunction.h"

IdentityFunction::IdentityFunction(int numInputs)
{
	this->numInputs = numInputs;
	this->numOutputs = numInputs;
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

void IdentityFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	ImVec2 l_start(origin.x - (DRAW_LEN * scale), origin.y + (DRAW_LEN * scale));
	ImVec2 l_end(origin.x + (DRAW_LEN * scale), origin.y - (DRAW_LEN * scale));

	canvas->AddLine(l_start, l_end, BLACK);
}