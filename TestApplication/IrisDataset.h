#pragma once

#include <Eigen/Core>
#include <iostream>
#include <fstream>

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

		columnNames = new string[colCount];

		// Header containing the column names
		getline(irisFile, line);
		size_t pos = 0;
		size_t last = 0;
		string token = "";
		for (int i = 0; i < colCount; i++)
		{
			pos = line.find(",", last);
			token = line.substr(last, (pos-last));
			columnNames[i] = token;
			last = pos + 1;
		}

		const int featureCount = 4;
		features = MatrixXd(rowCount, featureCount);
		labels = MatrixXd(rowCount, 1);
		int c = -1;
		for (int i = 0; i < rowCount; i++)
		{
			c = (i % 50 == 0) ? c+1 : c;
			int k = i % 50;
			int index = (k * 3 + c) % 150;
			getline(irisFile, line);
			last = 0;
			for (int j = 0; j < featureCount; j++)
			{
				pos = line.find(",", last);
				token = line.substr(last, (pos - last));
				features(index, j) = stod(token); // TODO
				last = pos + 1;
			}
			pos = line.find(",", last);
			token = line.substr(last, (pos - last));
			labels(index, 0) = getIrisSpeciesNum(token); // TODO
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

	MatrixXd getFeatures()
	{
		return features;
	}

	MatrixXd getLabels()
	{
		return labels;
	}

	MatrixXd getLabelsOneHot()
	{
		int r = labels.rows();
		MatrixXd oneHot = MatrixXd::Zero(r, 3);

		for (int i = 0; i < r; i++)
		{
			oneHot(i, labels(i)) = 1.0;
		}

		return oneHot;
	}

private:
	string* columnNames;
	const int rowCount = 150; // Excluding header
	const int colCount = 5;

	MatrixXd features;
	MatrixXd labels;
};