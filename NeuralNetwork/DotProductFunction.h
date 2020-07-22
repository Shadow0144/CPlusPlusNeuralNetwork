#pragma once

#include "Function.h"

class DotProductFunction : public Function
{
public:
	DotProductFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	int numOutputs();
};