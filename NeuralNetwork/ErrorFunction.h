#pragma once

#pragma warning(push, 0)
#include <Eigen/Core>
#pragma warning(pop)

using namespace Eigen;

class ErrorFunction
{
public:
	virtual double getError(MatrixXd predicted, MatrixXd actual) = 0;
	virtual MatrixXd getDerivativeOfError(MatrixXd predicted, MatrixXd actual) = 0;
};