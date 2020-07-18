#pragma once

#pragma warning(push, 0)
#include <Eigen/Core>
#pragma warning(pop)

#include "NetworkVisualizer.h"
#include "ParameterSet.h"

using namespace Eigen;

class Function
{
public:
	virtual MatrixXd feedForward(MatrixXd inputs) = 0;
	virtual MatrixXd backPropagate(MatrixXd lastInput, MatrixXd errors) = 0;
	double applyBackProgate(); // Returns the sum of the change in the weights
	virtual bool hasBias() = 0;
	virtual void draw(NetworkVisualizer canvas);
	ParameterSet getWeights() { return weights; }
	virtual int numOutputs() = 0;
protected:
	int numInputs = 0;
	ParameterSet weights;
	const int DRAW_LEN = 16;

	const double ALPHA = 0.1; // Learning rate

	friend class NetworkVisualizer;
};