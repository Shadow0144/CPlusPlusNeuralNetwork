#include "FunctionVisualizer.h"
#include "NetworkVisualizer.h"

FunctionVisualizer::FunctionVisualizer(NetworkVisualizer* visualizer, NeuralNetwork* network)
{
	this->visualizer = visualizer;
	this->network = network;
}

FunctionVisualizer::~FunctionVisualizer()
{

}

void FunctionVisualizer::draw(ImDrawList* canvas, const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
{
	// Calculate the drawing space parameters
	ImVec2 winSize = visualizer->getWindowSize();
	const double WIDTH = winSize.x;
	const double HEIGHT = winSize.y;
	const double HALF_WIDTH = WIDTH * 0.5;
	const double HALF_HEIGHT = HEIGHT * 0.5;

	// Draw the function approximated by the network
	const double BUFFER = 20.0;
	const double GRID_SIZE = 250.0;
	const double LEFT = WIDTH - BUFFER - GRID_SIZE;
	const double TOP = HEIGHT - BUFFER - GRID_SIZE;
	const double CENTER_X = LEFT + (GRID_SIZE * 0.5);
	const double CENTER_Y = TOP + (GRID_SIZE * 0.5);

	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);
	const ImColor DARK_GRAY(0.5f, 0.5f, 0.5f, 1.0f);
	const ImColor BLUE(0.0f, 0.0f, 1.0f, 1.0f);
	const ImColor RED(1.0f, 0.0f, 0.0f, 1.0f);

	const double EDGE_BUFFER = 1.0;
	const double PREDICTED_RADIUS = 0.5;
	const double TARGET_RADIUS = 0.5;

	const double PEG_LENGTH = 2;
	const int PEG_COUNT = 10 + 1;

	canvas->AddRectFilled(ImVec2(LEFT, TOP), ImVec2(LEFT + GRID_SIZE, TOP + GRID_SIZE), WHITE);

	//const int DOT_LENGTH = 4;
	ImVec2 zero_x_left(LEFT, CENTER_Y);
	ImVec2 zero_x_right(LEFT + GRID_SIZE, CENTER_Y);
	ImVec2 zero_y_bottom(CENTER_X, TOP);
	ImVec2 zero_y_top(CENTER_X, TOP + GRID_SIZE);
	canvas->AddLine(zero_x_left, zero_x_right, DARK_GRAY);
	canvas->AddLine(zero_y_bottom, zero_y_top, DARK_GRAY);

	const double PEG_SPACING = GRID_SIZE / (PEG_COUNT-1);
	for (int i = 0; i < PEG_COUNT; i++)
	{
		ImVec2 x_bottom(LEFT + (i * PEG_SPACING), CENTER_Y + PEG_LENGTH);
		ImVec2 x_top(LEFT + (i * PEG_SPACING), CENTER_Y - PEG_LENGTH);
		ImVec2 y_left(CENTER_X - PEG_LENGTH, TOP + (i * PEG_SPACING));
		ImVec2 y_right(CENTER_X + PEG_LENGTH, TOP + (i * PEG_SPACING));
		canvas->AddLine(x_bottom, x_top, DARK_GRAY);
		canvas->AddLine(y_left, y_right, DARK_GRAY);
	}

	const int N = inputs.shape()[0];
	const double STEP_SIZE = 0.1;
	auto inputShape = inputs.shape();
	inputShape[0] = ((int)(2.0 / STEP_SIZE));
	xt::xstrided_slice_vector inputsSV({ xt::all() });
	for (int i = 1; i < inputShape.size(); i++)
	{
		inputsSV.push_back(0);
	}
	xt::xstrided_slice_vector outputsSV({ xt::all() });
	for (int i = 1; i < targets.dimension(); i++)
	{
		outputsSV.push_back(0);
	}

	double dist = max((xt::strided_view(inputs, inputsSV)(N - 1) - xt::strided_view(inputs, inputsSV)(0)), 2.0);
	double w = dist * EDGE_BUFFER;
	double rescale = (GRID_SIZE / w);

	// Fix: Clipping y
	xt::xarray<double> xs = xt::zeros<double>(inputShape);
	xt::strided_view(xs, inputsSV) = xt::linspace<double>(-1.0, 1.0, inputShape[0]);
	auto ys = network->predict(xs);
	for (int i = 0; i < (inputShape[0] - 1); i++)
	{
		float x1 = (rescale * xt::strided_view(xs, inputsSV)(i));
		float x2 = (rescale * xt::strided_view(xs, inputsSV)(i + 1));
		float y1 = rescale * xt::strided_view(ys, outputsSV)(i);
		float y2 = rescale * xt::strided_view(ys, outputsSV)(i + 1);
		ImVec2 l_start(CENTER_X + x1, CENTER_Y - y1);
		ImVec2 l_end(CENTER_X + x2, CENTER_Y - y2);
		canvas->AddLine(l_start, l_end, BLUE, PREDICTED_RADIUS);
	}

	for (int i = 0; i < N; i++)
	{
		canvas->AddCircleFilled(
			ImVec2(CENTER_X + (rescale * xt::strided_view(inputs, inputsSV)(i)),
					CENTER_Y - (rescale * xt::strided_view(targets, outputsSV)(i))),
					TARGET_RADIUS, RED);
	}
}