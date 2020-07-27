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
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	double getK();
	void setK(double k);

private:
	MatrixXd lastOutput;

	double k = 1.0; // Sharpness coefficient
};