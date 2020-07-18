#include "MSEFunction.h"

#include <unsupported/Eigen/MatrixFunctions>

double MSEFunction::getError(MatrixXd predicted, MatrixXd actual)
{
	int n = predicted.rows();
	int m = predicted.cols();
	MatrixXd errors = predicted - actual;
	errors = errors.array().pow(2);
	errors /= (n * m);
	double error = errors.sum();
	return error;
}

MatrixXd MSEFunction::getDerivativeOfError(MatrixXd predicted, MatrixXd actual)
{
	int n = predicted.rows();
	int m = predicted.cols();
	MatrixXd sum = (predicted - actual).colwise().sum(); // Sum across the columns giving one error per example (row)
	MatrixXd errors = (2.0 / (n * m)) * sum;
	return errors;
}