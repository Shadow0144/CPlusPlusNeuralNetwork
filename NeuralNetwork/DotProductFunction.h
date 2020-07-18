#pragma once

#include "Function.h"

class DotProductFunction : public Function
{
public:
	DotProductFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	void draw(NetworkVisualizer canvas);

	int numOutputs();
};