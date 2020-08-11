#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#include <iostream>
#include <fstream>
#pragma warning(pop)

using namespace std;

enum class IrisSpecies
{
	setosa = 0,
	versicolor = 1,
	virginica = 2
};

class IrisDataset
{
public:
	IrisDataset()
	{
		ifstream irisFile;
		irisFile.open("iris.dat");
		string line;

		columnNames = new string[COL_COUNT];

		// Header containing the column names
		getline(irisFile, line);
		size_t pos = 0;
		size_t last = 0;
		string token = "";
		for (int i = 0; i < COL_COUNT; i++)
		{
			pos = line.find(",", last);
			token = line.substr(last, (pos-last));
			columnNames[i] = token;
			last = pos + 1;
		}

		const int FEATURE_COUNT = 4;
		xt::xarray<int>::shape_type shapeFeatures = { ROW_COUNT, FEATURE_COUNT };
		features = xt::xarray<double>(shapeFeatures);
		xt::xarray<int>::shape_type shapeLabels = { ROW_COUNT, 1 };
		labels = xt::xarray<double>(shapeLabels);
		int c = -1;
		for (int i = 0; i < ROW_COUNT; i++)
		{
			c = (i % 50 == 0) ? c+1 : c;
			int k = i % 50;
			int index = (k * 3 + c) % 150;
			getline(irisFile, line);
			last = 0;
			for (int j = 0; j < FEATURE_COUNT; j++)
			{
				pos = line.find(",", last);
				token = line.substr(last, (pos - last));
				features(index, j) = stod(token);
				last = pos + 1;
			}
			pos = line.find(",", last);
			token = line.substr(last, (pos - last));
			labels(index, 0) = getIrisSpeciesNum(token);
		}

		irisFile.close();
	};

	~IrisDataset()
	{
		delete[] columnNames;
	}

	string getIrisSpeciesName(IrisSpecies species)
	{
		return getIrisSpeciesName(species);
	}

	string getIrisSpeciesName(int species)
	{
		string speciesString = "";
		switch (species)
		{
			case 0: //IrisSpecies::setosa:
				speciesString = "setosa";
				break;
			case 1: //IrisSpecies::versicolor:
				speciesString = "versicolor";
				break;
			case 2: //IrisSpecies::virginica:
				speciesString = "virginica";
				break;
		}
		return speciesString;
	}

	int getIrisSpeciesNum(string speciesString)
	{
		int species = -1;
		if (speciesString.compare("setosa") == 0)
		{
			species = 0;
		}
		else if (speciesString.compare("versicolor") == 0)
		{
			species = 1;
		}
		else if (speciesString.compare("virginica") == 0)
		{
			species = 2;
		}
		else { }
		return species;
	}

	xt::xarray<double> getFeatures()
	{
		return features;
	}

	xt::xarray<double> getLabels()
	{
		return labels;
	}

	xt::xarray<double> getLabelsOneHot()
	{
		xt::xarray<int>::shape_type labelShape = { ROW_COUNT, 3 };
		xt::xarray<double> oneHot = xt::zeros<double>(labelShape);

		for (int i = 0; i < ROW_COUNT; i++)
		{
			oneHot(i, labels(i)) = 1.0;
		}

		return oneHot;
	}

private:
	string* columnNames;
	const size_t ROW_COUNT = 150; // Excluding header
	const size_t COL_COUNT = 5;

	xt::xarray<double> features;
	xt::xarray<double> labels;
};