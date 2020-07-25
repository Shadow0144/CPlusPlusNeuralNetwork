#pragma once

#include "Function.h"

class SoftmaxFunction : public Function
{
public:
	SoftmaxFunction(int numInputs, int numOutputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	int getNumOutputs();

private:
	MatrixXd lastOutput;
	int numOutputs;
};