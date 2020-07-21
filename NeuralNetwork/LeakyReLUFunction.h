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
	virtual void draw(ImDrawList* canvas, ImVec2 origin, float scale);

	float getA();
	void setA(float a);

	int numOutputs();

private:
	MatrixXd lastOutput;

	float a = 0.01f; // Leak coefficient
};