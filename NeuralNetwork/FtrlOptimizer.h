#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "ErrorFunction.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

// Follow The Regularized Leader (-Proximal) optimizer
class FtrlOptimizer : public Optimizer
{
public:
	FtrlOptimizer(std::vector<NeuralLayer*>* layers, int batchSize = -1, double alpha = 1.0, double beta = 1.0, double lamda1 = 0.001, double lamda2 = 0.001);
	FtrlOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	double backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets); // Single step
	xt::xarray<double> getDeltaWeight(long parameterID, const xt::xarray<double>& gradient); // Adjusts the gradient based on the optimizer

	void substituteParameters(ParameterSet& parameterSet);

	const static std::string BATCH_SIZE; // = "batchSize"; // Parameter string [OPTIONAL] // Values less than 0 for batch size = N
	const static std::string ALPHA; // = "alpha"; // Parameter string [REQUIRED] // 
	const static std::string BETA; // = "beta"; // Parameter string [REQUIRED] // 
	const static std::string LAMDA1; // = "lamda1"; // Parameter string [OPTIONAL] // 
	const static std::string LAMDA2; // = "lamda2"; // Parameter string [OPTIONAL] // 

private:
	double alpha; // 
	double beta; // 
	double lamda1; // 
	double lamda2; // 

	std::map<long, xt::xarray<double>> z; // 
	std::map<long, xt::xarray<double>> n; // 
	std::map<long, xt::xarray<double>> w; // Current weights
};