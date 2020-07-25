#pragma once

#include "Function.h"

// Leaky ReLU / Parametric ReLU
class LeakyReLUFunction : public Function
{
public:
	LeakyReLUFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	double getA();
	void setA(double a);

	int getNumOutputs();

private:
	MatrixXd lastOutput;

	double a = 0.01; // Leak coefficient
};