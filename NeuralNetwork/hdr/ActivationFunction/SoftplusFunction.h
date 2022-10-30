#pragma once

#include "ActivationFunction.h"

// Softplus / Smooth Rectified Linear Unit
class SoftplusFunction : public ActivationFunction
{
public:
	SoftplusFunction();
	SoftplusFunction(double k);

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

	double getParameter(const std::string& parameterName) const;
	void setParameter(const std::string& parameterName, double value);

	void saveParameters(std::string fileName);
	void loadParameters(std::string fileName);

	double getK() const;
	void setK(double k);
	const static std::string K; // = "k"; // Parameter string [OPTIONAL]

private:
	double activate(double z) const;

	double softplus(double z) const;
	xt::xarray<double> softplus(const xt::xarray<double>& z) const;

	double k = 1.0; // Sharpness coefficient
};