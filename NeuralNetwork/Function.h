#pragma once

#pragma warning(push, 0)
#include <Eigen/Core>
#include <Eigen/Dense>
#pragma warning(pop)

#include "imgui.h"
#include "ParameterSet.h"

using namespace Eigen;

class Function
{
public:
	virtual MatrixXd feedForward(MatrixXd inputs) = 0;
	virtual MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors) = 0;
	double applyBackProgate(); // Returns the sum of the change in the weights
	virtual bool hasBias() = 0;
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);
	ParameterSet getWeights() { return weights; }
	virtual int numOutputs() = 0;
protected:
	int numInputs = 0;
	ParameterSet weights;
	const double DRAW_LEN = 16;

	MatrixXd approximateBezier(MatrixXd points);

	const double ALPHA = 0.1; // Learning rate
};