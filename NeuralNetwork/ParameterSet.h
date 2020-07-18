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
	MatrixXd getDeltaParameters();
	void setDeltaParameters(MatrixXd deltaParameters);
	void applyDeltaParameters();

private:
	MatrixXd parameters;
	MatrixXd deltaParameters;
};