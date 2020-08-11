#include "SoftmaxFunction.h"

#include <iostream>

using namespace std;

SoftmaxFunction::SoftmaxFunction(std::vector<size_t> numInputs, std::vector<size_t> numOutputs)
{
	/*this->hasBias = true;
	this->inputDims = numInputs.size();
	this->numInputs = std::vector<size_t>(numInputs);
	this->numOutputs = std::vector<size_t>(numOutputs);
	this->weights.setParametersRandom(numInputs);
	for (int i = 0; i < inputDims - 1; i++)
	{
		this->sumIndices.push_back(i);
		this->biaslessView.push_back(xt::all());
	}
	this->biaslessView.push_back(xt::range(0, numInputs[inputDims - 1] - 1));
	this->lastInput = xt::ones<double>(numInputs);*/
}

xt::xarray<double> SoftmaxFunction::feedForward(xt::xarray<double> inputs)
{
	//xt::strided_view(lastInput, biaslessView) = inputs;
	/*double total = 0.0;

	lastOutput = MatrixXd(1, numOutputs);
	double c = -inputs.maxCoeff();
	for (int i = 0; i < numOutputs; i++)
	{
		MatrixXd z = inputs * weights.getParameters().col(i);
		lastOutput(i) = exp(z(0) + c);
		total += lastOutput(i);
	}
	lastOutput /= total;

	return lastOutput;*/
	return inputs;
}

xt::xarray<double> SoftmaxFunction::backPropagate(xt::xarray<double> errors)
{
	/*MatrixXd outputDiagonal = lastOutput.row(0).asDiagonal();
	MatrixXd lastOutputRep = lastOutput.replicate(numOutputs, 1);
	MatrixXd prime = -lastOutputRep.cwiseProduct(MatrixXd::Identity(numOutputs, numOutputs) - lastOutputRep.transpose());
	MatrixXd sigma = errors * prime;

	weights.incrementDeltaParameters(-ALPHA * lastInput.transpose() * sigma);

	// Strip away the bias parameter and weight the sigma by the incoming weights
	MatrixXd weightsPrime = weights.getParameters().block(0, 0, (numInputs - 1), numOutputs);

	return weightsPrime * sigma.transpose();*/
	return errors;
}

void SoftmaxFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	int r = 3;
	double range = 3.0;

	int resolution = (r * 4) + 1;
	/*MatrixXd simInput = MatrixXd::Zero(1, numInputs);
	simInput(0, numInputs - 1) = 1; // Bias
	MatrixXd sP(resolution, 2);
	for (int r = 0; r < resolution; r++)
	{
		// TODO: Clip if exceeds range
		sP(r, 0) = range * (2.0 * r) / (resolution - 1.0) - range;
		simInput(0, 0) = sP(r, 0);
		sP(r, 1) = feedForward(simInput)(0);
	}

	float rescale = (1.0 / range) * DRAW_LEN * scale;
	for (int d = 0; d < (resolution - 3); d += 3)
	{
		MatrixXd points = approximateBezier(sP.block(d, 0, 4, 2));
		canvas->AddBezierCurve(
			ImVec2(origin.x + (points(0, 0) * rescale), origin.y - (points(0, 1) * rescale)),
			ImVec2(origin.x + (points(1, 0) * rescale), origin.y - (points(1, 1) * rescale)),
			ImVec2(origin.x + (points(2, 0) * rescale), origin.y - (points(2, 1) * rescale)),
			ImVec2(origin.x + (points(3, 0) * rescale), origin.y - (points(3, 1) * rescale)),
			BLACK, 1);
	}*/
}