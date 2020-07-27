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
	void setParametersRandom(int inputCount);
	void setParametersZero(int inputCount);
	void setParametersOne(int inputCount);
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