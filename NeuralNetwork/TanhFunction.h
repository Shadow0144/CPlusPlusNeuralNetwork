#pragma once

#include "Function.h"

class TanhFunction : public Function
{
public:
	TanhFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	void draw(NetworkVisualizer canvas);

	int numOutputs();

private:
	MatrixXd lastOutput;
};