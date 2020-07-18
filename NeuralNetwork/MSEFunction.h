#pragma once

#include "ErrorFunction.h"

class MSEFunction : public ErrorFunction
{
public:
	double getError(MatrixXd predicted, MatrixXd actual);
	MatrixXd getDerivativeOfError(MatrixXd predicted, MatrixXd actual);
};