#include "CrossEntropyFunction.h"

#include <unsupported/Eigen/MatrixFunctions>

const double c = pow(10, -8);

double CrossEntropyFunction::getError(MatrixXd predicted, MatrixXd actual)
{
	int n = predicted.rows();
	MatrixXd errors = actual * MatrixXd(log(predicted.transpose().array() + c));
	double error = errors.sum() / n;
	return error;
}

MatrixXd CrossEntropyFunction::getDerivativeOfError(MatrixXd predicted, MatrixXd actual)
{
	MatrixXd errors = (predicted - actual);
	return errors;
}