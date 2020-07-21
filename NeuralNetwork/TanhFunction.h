#pragma once

#include "Function.h"

class TanhFunction : public Function
{
public:
	TanhFunction(int numInputs);

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors);
	bool hasBias();
	virtual void draw(ImDrawList* canvas, ImVec2 origin, float scale);

	int numOutputs();

private:
	MatrixXd lastOutput;
};