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
	xt::xarray<double> getParameters();
	void setParameters(xt::xarray<double> parameters);
	void setParametersRandom(size_t numParameters);
	void setParametersRandom(std::vector<size_t> numParameters);
	void setParametersZero(size_t numParameters);
	void setParametersZero(std::vector<size_t> numParameters);
	void setParametersOne(size_t numParameters);
	void setParametersOne(std::vector<size_t> numParameters);
	xt::xarray<double> getDeltaParameters();
	void incrementDeltaParameters(xt::xarray<double> deltaParameters);
	void applyDeltaParameters();

private:
	mutable std::shared_mutex weightsMutex;
	xt::xarray<double> parameters;
	xt::xarray<double> deltaParameters;
	int batchSize;
};