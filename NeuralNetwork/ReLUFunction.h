#pragma once

#include "Function.h"

class ReLUFunction : public Function
{
public:
	ReLUFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	void draw(NetworkVisualizer canvas);

	int numOutputs();

private:
	MatrixXd lastOutput;
};