#pragma once

#include <exception>

class InputLayerConfigurationException : public std::exception
{
	const char* what() const throw()
	{
		return "An input layer must be and can only be the first layer of a network.";
	}
};

class NeuralLayerConfigurationException : public std::exception
{
	const char* what() const throw()
	{
		return "This layer is missing a parent. Ensure that there is an input layer at the beginning of your network.";
	}
};

class NeuralLayerInputShapeException : public std::exception
{
	const char* what() const throw()
	{
		return "The input shape to this layer is incompatible with this layer.";
	}
};

class NeuralLayerConvolutionShapeException : public std::exception
{
	const char* what() const throw()
	{
		return "Parameter convolutionShape has an incompatible shape with this layer.";
	}
};

class NeuralLayerPoolingShapeException : public std::exception
{
	const char* what() const throw()
	{
		return "Parameter filterShape has an incompatible shape with this layer.";
	}
};

class NeuralLayerSqueezeShapeException : public std::exception
{
	const char* what() const throw()
	{
		return "An incompatible dimension list was passed. Only dimensions of size 0 or 1 can be removed by squeezing, and the list cannot be longer than the input dimensionality.";
	}
};