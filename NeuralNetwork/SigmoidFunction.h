#pragma once

#include "Function.h"

class SigmoidFunction : public Function
{
public:
	SigmoidFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	int numOutputs();

private:
	MatrixXd lastOutput;
};