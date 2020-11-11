#include "MaxPooling2DFunction.h"
#include "NeuralLayer.h"

#include "Test.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#pragma warning(pop)

using namespace std;

// TODO: Padding and dimensions
MaxPooling2DFunction::MaxPooling2DFunction(std::vector<size_t> filterShape)
{
	this->hasBias = false;
	this->filterShape = filterShape;
}

xt::xarray<double> MaxPooling2DFunction::feedForward(xt::xarray<double> inputs)
{
	lastInput = inputs;

	const int DIMS = lastInput.dimension();
	const int DIM1 = DIMS - 3; // First dimension
	const int DIM2 = DIMS - 2; // Second dimension
	const int DIMC = DIMS - 1; // Channels
	auto shape = lastInput.shape();
	auto maxShape = xt::svector<size_t>(shape);
	lastInputMask = xt::xarray<double>(shape); // Same shape as the input
	shape[DIM1] = ceil(shape[DIM1] / filterShape[0]);
	shape[DIM2] = ceil(shape[DIM2] / filterShape[1]);
	xt::xarray<double> output = xt::xarray<double>(shape);
	maxShape[DIM1] = 1;
	maxShape[DIM2] = 1;

	xt::xstrided_slice_vector inputWindowView;
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f <= DIMC; f++)
	{
		inputWindowView.push_back(xt::all());
		outputWindowView.push_back(xt::all());
	}

	int k = 0;
	int l = 0;
	const int I = inputs.shape()[DIM1];
	const int J = inputs.shape()[DIM2];
	for (int i = 0; i < I; i += filterShape[0])
	{
		inputWindowView[DIM1] = xt::range(i, i + filterShape[0]);
		outputWindowView[DIM1] = k++; // Increment after assignment
		for (int j = 0; j < J; j += filterShape[1])
		{
			inputWindowView[DIM2] = xt::range(j, j + filterShape[1]);
			outputWindowView[DIM2] = l++; // Increment after assignment
			auto window = xt::xarray<double>(xt::strided_view(inputs, inputWindowView));
			auto maxes = xt::amax(window, { DIM1, DIM2 });
			auto maxesComp = xt::xarray<double>(maxes);
			maxesComp.reshape(maxShape);
			//if (false)
			{
				/*std::cout << "M: " << maxes.dimension() << ", " << maxes.shape()[0] << " x " << 
					maxes.shape()[1] << " x " << maxes.shape()[2] << " x " << maxes.shape()[3] << endl;
				std::cout << "W: " << window.dimension() << ", " << window.shape()[0] << " x " <<
					window.shape()[1] << " x " << window.shape()[2] << " x " << window.shape()[3] << endl;
				std::cout << "I: " << xt::strided_view(lastInputMask, inputWindowView).dimension() << ", " <<
					xt::strided_view(lastInputMask, inputWindowView).shape()[0] << " x " << 
					xt::strided_view(lastInputMask, inputWindowView).shape()[1] << " x " << 
					xt::strided_view(lastInputMask, inputWindowView).shape()[2] << " x " << 
					xt::strided_view(lastInputMask, inputWindowView).shape()[3] << endl;
				auto t = xt::equal(window, maxes);
				std::cout << window(0, 0, 0, 0) << " " << window(0, 0, 1, 0) << endl << window(0, 1, 0, 0) << " " << window(0, 1, 1, 0) << endl;
				std::cout << maxes(0, 0, 0, 0) << endl;
				std::cout << t(0, 0, 0, 0) << " " << t(0, 0, 1, 0) << endl << t(0, 1, 0, 0) << " " << t(0, 1, 1, 0) << endl;
				std::cout << "O: " << xt::where(xt::equal(window, maxes), 1, 0).dimension() << ", " <<
					xt::where(xt::equal(window, maxes), 1, 0).shape()[0] << " x " <<
					xt::where(xt::equal(window, maxes), 1, 0).shape()[1] << " x " <<
					xt::where(xt::equal(window, maxes), 1, 0).shape()[2] << " x " <<
					xt::where(xt::equal(window, maxes), 1, 0).shape()[3] << endl;*/
				/*auto t = xt::strided_view(output, outputWindowView);
				std::cout << "T: " << t.dimension() << ", " <<
					t.shape()[0] << " x " <<
					t.shape()[1] << " x " <<
					t.shape()[2] << " x " <<
					t.shape()[3] << endl;*/
			}
			xt::strided_view(lastInputMask, inputWindowView) = xt::equal(window, maxesComp);
			xt::strided_view(output, outputWindowView) = maxes;
		}
		l = 0;
	}

	//cout << "Pool: " << xt::sum(output)(0) << endl;

	return output;
}

xt::xarray<double> MaxPooling2DFunction::backPropagate(xt::xarray<double> sigmas)
{
	// Reverse what was done in feedforward, the input is now the output
	const int DIMS = lastInput.dimension();
	const int DIM1 = DIMS - 3; // First dimension
	const int DIM2 = DIMS - 2; // Second dimension
	const int DIMC = DIMS - 1; // Channels
	auto shape = lastInput.shape();
	shape[DIM1] = ceil(shape[DIM1] / filterShape[0]);
	shape[DIM2] = ceil(shape[DIM2] / filterShape[1]);
	auto sigmaShape = sigmas.shape();
	sigmaShape[DIM1] = 1;
	sigmaShape[DIM2] = 1;

	//print_dims(sigmas);
	//for (int i = 0; i < sigmas.shape()[0]; i++) // N
	//{
	//	for (int j = 0; j < sigmas.shape()[1]; j++) // x
	//	{
	//		for (int k = 0; k < sigmas.shape()[2]; k++) // y
	//		{
	//			for (int l = 0; l < sigmas.shape()[3]; l++) // c
	//			{
	//				if (abs(sigmas(i, j, k, l)) > 0.0001)
	//				{
	//					cout << sigmas(i, j, k, l) << ", ";
	//				}
	//				else
	//				{
	//					cout << 0 << ", ";
	//				}
	//			}
	//			cout << "] ";
	//		}
	//		cout << "} ";
	//	}
	//	cout << endl;
	//}

	xt::xarray<double> sigmasPrime = xt::xarray<double>(lastInput.shape());

	xt::xstrided_slice_vector primeWindowView;
	xt::xstrided_slice_vector sigmaWindowView;
	for (int f = 0; f <= DIMC; f++)
	{
		primeWindowView.push_back(xt::all());
		sigmaWindowView.push_back(xt::all());
	}

	int k = 0;
	int l = 0;
	const int I = shape[DIM1];
	const int J = shape[DIM2];
	for (int i = 0; i < I; i++)
	{
		primeWindowView[DIM1] = xt::range(i, i + filterShape[0]);
		sigmaWindowView[DIM1] = k++; // Increment after assignment
		for (int j = 0; j < J; j++)
		{
			primeWindowView[DIM2] = xt::range(j, j + filterShape[1]);
			sigmaWindowView[DIM2] = l++; // Increment after assignment
			auto window = xt::strided_view(lastInputMask, primeWindowView);
			auto sigma = xt::xarray<double>(xt::strided_view(sigmas, sigmaWindowView));
			sigma.reshape(sigmaShape);
			/*cout << "W: " << window.dimension() << ", " << window.shape()[0] << " x "
				<< window.shape()[1] << " x " << window.shape()[2] << " x " << window.shape()[3] << endl;
			cout << "S: " << sigma.dimension() << ", " << sigma.shape()[0] << " x "
				<< sigma.shape()[1] << " x " << sigma.shape()[2] << " x " << sigma.shape()[3] << endl;*/
			xt::strided_view(sigmasPrime, primeWindowView) = window * sigma;
		}
		l = 0;
	}
	//cout << "Prime: " << xt::sum(sigmasPrime)(0) << endl;

	return sigmasPrime;
}

void MaxPooling2DFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = weights.getParameters()(0, i);
		double inv_slope = 1.0 / abs(slope);
		double x1, x2, y1, y2;
		if (slope > 0.0)
		{
			x1 = -1.0;
			x2 = +min(1.0, inv_slope);
			y1 = 0.0;
			y2 = (x2 * slope);
		}
		else
		{
			x1 = -min(1.0, inv_slope);
			x2 = 1.0;
			y1 = (x1 * slope);
			y2 = 0.0;
		}

		ImVec2 l_start(position.x + (DRAW_LEN * x1 * scale), position.y - (DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (DRAW_LEN * x2 * scale), position.y - (DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}