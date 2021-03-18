#pragma once

namespace NeuralNetworkFileHelper
{
	inline bool fileExists(const std::string& name)
	{
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	};
}