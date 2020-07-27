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
	void setParametersRandom(int inputCount, int outputCount = 1);
	void setParametersZero(int inputCount, int outputCount = 1);
	void setParametersOne(int inputCount, int outputCount = 1);
	MatrixXd getDeltaParameters();
	void incrementDeltaParameters(MatrixXd deltaParameters);
	void applyDeltaParameters();

private:
	MatrixXd parameters;
	MatrixXd deltaParameters;
	int batchSize;
};