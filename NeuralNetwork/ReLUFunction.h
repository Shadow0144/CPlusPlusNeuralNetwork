#pragma once

#include "Function.h"

class ReLUFunction : public Function
{
public:
	ReLUFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	MatrixXd lastOutput;
};