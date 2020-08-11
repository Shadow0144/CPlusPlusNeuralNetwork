#include "Function.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

using namespace std;

double Function::applyBackProgate()
{
	double deltaWeight = xt::sum(xt::abs(weights.getDeltaParameters()))();
	weights.applyDeltaParameters();
	return deltaWeight; // Return the sum of how much the parameters have changed
}

xt::xarray<double> Function::dotProduct(xt::xarray<double> inputs)
{
	if (hasBias)
	{
		size_t inputDims = inputs.dimension();
		auto inputShape = inputs.shape();
		inputShape.at(inputDims - 1)++;
		xt::xstrided_slice_vector biaslessView;
		for (int i = 0; i < (inputDims - 1); i++)
		{
			biaslessView.push_back(xt::all());
		}
		biaslessView.push_back(xt::range(0, (numInputs - 1)));

		lastInput = xt::ones<double>(inputShape);
		xt::strided_view(lastInput, biaslessView) = inputs;
	}
	else
	{
		lastInput = inputs;
	}

	return xt::linalg::tensordot(lastInput, weights.getParameters(), 1); // The last dimension of the input with the first dimension of the weights
}

xt::xarray<double> Function::activationDerivative()
{
	return xt::xarray<double>();
}

xt::xarray<double> Function::denseBackpropagate(xt::xarray<double> sigmas)
{
	//cout << "Input dimension: " << lastInput.dimension() << " shape: " << lastInput.shape()[0] << ", " << lastInput.shape()[1] << endl;
	//cout << "Sigma dimension: " << sigmas.dimension() << " shape: " << sigmas.shape()[0] << ", " << sigmas.shape()[1] << endl;
	auto delta = xt::linalg::tensordot(xt::transpose(lastInput), sigmas, 1);

	weights.incrementDeltaParameters(-ALPHA * delta);
	auto biaslessWeights = xt::view(weights.getParameters(), xt::range(0, (numInputs - 1)), xt::all());

	//cout << "Weights dimension: " << biaslessWeights.dimension() << " shape: " << biaslessWeights.shape()[0] << ", " << biaslessWeights.shape()[1] << endl;
	auto newSigmas = xt::linalg::tensordot(sigmas, xt::transpose(biaslessWeights), 1); // The last {1} axes of errors and the first {1} axes of the weights transposed
	//cout << "New sigma dimension: " << newSigmas.dimension() << " shape: " << newSigmas.shape()[0] << ", " << newSigmas.shape()[1] << endl;

	//cout << endl;

	return newSigmas;
}

MatrixXd Function::approximateBezier(MatrixXd points)
{
	//int pCount = points.rows();
	//int k = pCount - 1;

	/*MatrixXd T(pCount, pCount);
	for (int i = 0; i < pCount; i++)
	{
		for (int j = 0; j < pCount; j++)
		{
			double jk = ((double)j) / ((double)k);
			T(k-i, k-j) = pow(jk, i);
		}
	}*/

	MatrixXd T(4, 4);
	T << 1.0,       0.0,       0.0,        0.0,
		 1.0, 1.0 / 3.0, 1.0 / 9.0, 1.0 / 27.0,
		 1.0, 2.0 / 3.0, 4.0 / 9.0, 8.0 / 27.0,
		 1.0,       1.0,       1.0,        1.0;

	MatrixXd M(4, 4);
	M << 1,  0,  0,  0,
		-3,  3,  0,  0,
		 3, -6,  3,  0,
		-1,  3, -3,  1;

	/*MatrixXd M(7, 7);
	M <<     1,   0,   0,   0,   0,   0,   0,
			 -6,   6,   0,   0,   0,   0,   0,
			 15, -30,  15,   0,   0,   0,   0,
			-20,  60, -60, -20,   0,   0,   0,
			 15, -60,  90, -60,  15,   0,   0,
			 -6,  30, -60,  60, -30,   6,   0,
			  1,  -6,  15, -20,  15,  -6,   1;*/

	MatrixXd Mi = M.inverse();
	MatrixXd TT = T.transpose();
	MatrixXd A = Mi * (TT * T).inverse() * TT;

	return A * points;
}

void Function::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);
	const double DOT_LENGTH = 4;

	double rescale = DRAW_LEN * scale;

	ImVec2 start(origin.x - rescale, origin.y + rescale);
	ImVec2 end(origin.x + rescale, origin.y - rescale);

	canvas->AddRectFilled(start, end, WHITE);
	canvas->AddRect(start, end, BLACK);

	ImVec2 zero_x_left(origin.x - rescale, origin.y);
	ImVec2 zero_x_right(origin.x + rescale, origin.y);
	canvas->AddLine(zero_x_left, zero_x_right, LIGHT_GRAY);
	//LineIterator itX(canvas.canvas, zero_x_left, zero_x_right, LINE_8);
	ImVec2 zero_y_base(origin.x, origin.y + rescale);
	ImVec2 zero_y_top(origin.x, origin.y - rescale);
	canvas->AddLine(zero_y_base, zero_y_top, LIGHT_GRAY);
	//LineIterator itY(canvas.canvas, zero_y_base, zero_y_top, LINE_8);
	/*for (int i = 0; i < itX.count; i++, itX++, itY++)
	{
		if (i % DOT_LENGTH != 0)
		{
			(*itX)[0] = DARK_GRAY;
			(*itX)[1] = DARK_GRAY;
			(*itX)[2] = DARK_GRAY;
			(*itY)[0] = DARK_GRAY;
			(*itY)[1] = DARK_GRAY;
			(*itY)[2] = DARK_GRAY;
		}
		else { }
	}*/
}