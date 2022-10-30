#pragma once

#include "ActivationFunction.h"

// Leaky Rectified Linear Unit
class LeakyReLUFunction : public ActivationFunction
{
public:
	LeakyReLUFunction();
	LeakyReLUFunction(double a);

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

	double getParameter(const std::string& parameterName) const;
	void setParameter(const std::string& parameterName, double value);

	void saveParameters(std::string fileName);
	void loadParameters(std::string fileName);

	double getA() const;
	void setA(double a);
	const static std::string A; // = "a"; // Parameter string [OPTIONAL]

private:
	xt::xarray<double> leakyReLU(const xt::xarray<double>& z) const;

	double a = 0.01; // Leak coefficient
};