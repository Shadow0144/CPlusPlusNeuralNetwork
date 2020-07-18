#pragma once

#include "Function.h"

class SigmoidFunction : public Function
{
public:
	SigmoidFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	void draw(NetworkVisualizer canvas);

	int numOutputs();

private:
	MatrixXd lastOutput;
};