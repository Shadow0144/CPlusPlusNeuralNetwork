#pragma once

#include "Function.h"

class ConvolutionFunction : public Function
{
public:
	ConvolutionFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	void draw(NetworkVisualizer canvas);

	int numOutputs();
};