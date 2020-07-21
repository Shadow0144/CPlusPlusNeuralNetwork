#pragma once

#include "Function.h"

class ConvolutionFunction : public Function
{
public:
	ConvolutionFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	virtual void draw(ImDrawList* canvas, ImVec2 origin, float scale);

	int numOutputs();
};