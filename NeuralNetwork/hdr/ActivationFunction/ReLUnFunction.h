#pragma once

#include "ActivationFunction.h"

// Rectified Linear Unit - n
class ReLUnFunction : public ActivationFunction
{
public:
	ReLUnFunction();
	ReLUnFunction(double n);

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

	double getParameter(const std::string& parameterName) const;
	void setParameter(const std::string& parameterName, double value);

	void saveParameters(std::string fileName);
	void loadParameters(std::string fileName);

	double getN() const;
	void setN(double n);
	const static std::string N; // = "n"; // Parameter string [OPTIONAL]

private:
	double n = 1.0; // Activation limit

	xt::xarray<double> reLUn(const xt::xarray<double>& z) const;
};