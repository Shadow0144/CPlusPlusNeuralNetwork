#pragma once

#pragma warning( disable : 26451 )

#include "ParameterSet.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <xtensor/xarray.hpp>
#pragma warning(pop)

enum class ActivationFunctionType;

class ActivationFunction
{
public:
	virtual xt::xarray<double> feedForward(const xt::xarray<double>& inputs) = 0;
	virtual xt::xarray<double> feedForwardTrain(const xt::xarray<double>& inputs);
	virtual xt::xarray<double> getGradient(const xt::xarray<double>& sigmas) = 0;
	virtual void applyBackPropagate();
	virtual std::vector<size_t> getOutputShape(std::vector<size_t> outputShape);
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);
	virtual void drawConversion(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

protected:
	xt::xarray<double> lastInput;
	xt::xarray<double> lastOutput;
	bool drawAxes = true;

	virtual double activate(double z); // For drawing
	xt::xarray<double> approximateBezier(const xt::xarray<double>& points); // For drawing
	void approximateFunction(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights); // For drawing
};