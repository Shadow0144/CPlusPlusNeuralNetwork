#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#include <shared_mutex>
#pragma warning(pop)

class ParameterSet
{
public:
	ParameterSet();
	ParameterSet(const ParameterSet& parameterSet);
	long getID();
	xt::xarray<double> getParameters() const;
	void setParameters(const xt::xarray<double>& parameters);
	void setParametersRandom(size_t numParameters);
	void setParametersRandom(const std::vector<size_t>& numParameters);
	void setParametersPositiveRandom(size_t numParameters);
	void setParametersPositiveRandom(const std::vector<size_t>& numParameters);
	void setParametersZero(size_t numParameters);
	void setParametersZero(const std::vector<size_t>& numParameters);
	void setParametersOne(size_t numParameters);
	void setParametersOne(const std::vector<size_t>& numParameters);
	xt::xarray<double> getDeltaParameters() const;
	void setDeltaParameters(const xt::xarray<double>& deltaParameters);
	void applyDeltaParameters();

private:
	mutable std::shared_mutex weightsMutex;
	xt::xarray<double> parameters;
	xt::xarray<double> deltaParameters;

	long parameterID; // For use by the optimizer to match with previous gradients
	static long nextParameterID;
};