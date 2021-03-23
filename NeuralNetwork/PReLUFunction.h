#pragma once

#include "ActivationFunction.h"

// Parametric Rectified Linear Unit
class PReLUFunction : public ActivationFunction
{
public:
	PReLUFunction(int numUnits);
	PReLUFunction(int numUnits, const xt::xarray<double>& a);

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

	double getParameter(const std::string& parameterName) const;
	void setParameter(const std::string& parameterName, double value);

	void saveParameters(std::string fileName);
	void loadParameters(std::string fileName);

	// a is usually learned and does not necessarily need setting
	xt::xarray<double> getA() const;
	void setA(xt::xarray<double> a);
	const static std::string NUM_UNITS; // = "numUnits"; // Parameter string [REQUIRED]
	const static std::string A; // = "a"; // Parameter string [OPTIONAL] // The index of a must be appended, e.g. a0, a1, or a2

private:
	xt::xarray<double> PReLU(const xt::xarray<double>& z) const;
	
	ParameterSet a; // Leak coefficients
};