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
	xt::xarray<double> getParametersWithoutBias() const;
	void setParameters(const xt::xarray<double>& parameters, bool hasBias = false);
	void setParametersRandom(size_t numParameters, bool hasBias = false);
	void setParametersRandom(const std::vector<size_t>& numParameters, bool hasBias = false);
	void setParametersPositiveRandom(size_t numParameters, bool hasBias = false);
	void setParametersPositiveRandom(const std::vector<size_t>& numParameters, bool hasBias = false);
	void setParametersZero(size_t numParameters, bool hasBias = false);
	void setParametersZero(const std::vector<size_t>& numParameters, bool hasBias = false);
	void setParametersOne(size_t numParameters, bool hasBias = false);
	void setParametersOne(const std::vector<size_t>& numParameters, bool hasBias = false);

	xt::xarray<double> getDeltaParameters() const;
	void setDeltaParameters(const xt::xarray<double>& deltaParameters);
	void applyDeltaParameters();
	
	void setUnregularized();
	bool getUnregularized() const;
	double getRegularizationLoss(double lambda1, double lambda2) const;
	xt::xarray<double> getRegularizedGradient(double lambda1, double lambda2) const;

private:
	mutable std::shared_mutex weightsMutex;
	xt::xarray<double> parameters;
	xt::xarray<double> deltaParameters;

	bool hasBias; // Flag for regularization
	bool unregularized; // For parameters that do not get regularized

	long parameterID; // For use by the optimizer to match with previous gradients
	static long nextParameterID;
};