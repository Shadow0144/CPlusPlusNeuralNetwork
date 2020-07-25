#pragma once

#pragma warning(push, 0)
#include <Eigen/Core>
#pragma warning(pop)

using namespace Eigen;

class ParameterSet
{
public:
	ParameterSet();
	MatrixXd getParameters();
	void setParametersRandom(int parameterCount);
	void setParametersZero(int parameterCount);
	void setParametersOne(int parameterCount);
	void setParametersRandom(int inputCount, int outputCount);
	void setParametersZero(int inputCount, int outputCount);
	void setParametersOne(int inputCount, int outputCount);
	MatrixXd getDeltaParameters();
	void setDeltaParameters(MatrixXd deltaParameters);
	void applyDeltaParameters();

private:
	MatrixXd parameters;
	MatrixXd deltaParameters;
};