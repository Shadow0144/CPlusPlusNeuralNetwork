#pragma once

#include "Function.h"

// Softplus / SmoothReLU
class SoftplusFunction : public Function
{
public:
	SoftplusFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	void draw(NetworkVisualizer canvas);

	double getK();
	void setK(double k);

	int numOutputs();

private:
	MatrixXd lastOutput;

	double k = 1.0; // Sharpness coefficient
};