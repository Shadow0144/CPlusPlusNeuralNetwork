#pragma once

#include "ActivationFunction/ActivationFunction.h"

// Exponential Linear Unit
class ELUFunction : public ActivationFunction
{
public:
	ELUFunction();
	ELUFunction(double alpha);

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

	double getParameter(const std::string& parameterName) const;
	void setParameter(const std::string& parameterName, double value);

	void saveParameters(std::string fileName);
	void loadParameters(std::string fileName);

	double getAlpha() const;
	void setAlpha(double alpha);
	const static std::string ALPHA; // = "eta"; // Parameter string [OPTIONAL]

private:
	double activate(double z) const;

	double ELU(double z) const;
	xt::xarray<double> ELU(const xt::xarray<double>& z) const;

	double alpha = 0.2;
};