#pragma once

#pragma warning( disable : 26451 )

#include "ParameterSet.h"
#include "Optimizer.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <xtensor/xarray.hpp>
#pragma warning(pop)

enum class ActivationFunctionType
{
	Identity,
	ReLU,
	AbsoluteReLU,
	CReLU,
	ELU,
	SELU,
	GELU,
	LeakyReLU,
	PReLU,
	ReLU6,
	ReLUn,
	Softplus,
	Exponential,
	Quadratic,
	Sigmoid,
	Tanh,
	HardSigmoid,
	Softsign,
	Swish,
	Maxout
};

class ActivationFunction
{
public:
	virtual xt::xarray<double> feedForward(const xt::xarray<double>& inputs) const = 0;
	virtual xt::xarray<double> feedForwardTrain(const xt::xarray<double>& inputs);
	virtual xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer) = 0;
	virtual void applyBackPropagate(); // Updates the parameters

	virtual double getParameter(const std::string& parameterName) const;
	virtual void setParameter(const std::string& parameterName, double value);

	virtual void saveParameters(std::string fileName);
	virtual void loadParameters(std::string fileName);

	virtual std::vector<size_t> getOutputShape(std::vector<size_t> outputShape) const;

	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;
	virtual void drawConversion(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

protected:
	xt::xarray<double> lastInput;
	xt::xarray<double> lastOutput;
	bool drawAxes = true;

	virtual double activate(double z) const; // For drawing
	xt::xarray<double> approximateBezier(const xt::xarray<double>& points) const; // For drawing
	void approximateFunction(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const; // For drawing
};