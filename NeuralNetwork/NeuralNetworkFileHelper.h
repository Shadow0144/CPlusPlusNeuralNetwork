#pragma once

namespace NeuralNetworkFileHelper
{
	inline bool folderExists(const std::string& name)
	{
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0 && buffer.st_mode & S_IFDIR);
	};

	inline bool fileExists(const std::string& name)
	{
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	};
}