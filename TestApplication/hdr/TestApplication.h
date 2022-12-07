#pragma once

#define _USE_MATH_DEFINES
#define NOMINMAX

#ifdef __GNUC__
#define LINUX
#else
#define WINDOWS
#endif
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentFolder _getcwd
#else
#include <unistd.h>
#define GetCurrentFolder getcwd
#endif

#pragma warning(push, 0)
#include <iostream>
#include <math.h>
#include <cmath>
#include <map>
#include <windows.h> 
#include <filesystem>
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>
#include <opencv2/opencv.hpp>
#pragma warning(pop)

#include "NeuralNetwork.h"

//#include "LossFunction/MeanSquareErrorLossFunction.h"
//#include "LossFunction/CrossEntropyErrorLossFunction.h"

//#include "Optimizer/SGDOptimizer.h"
//#include "Optimizer/AdagradOptimizer.h"
//#include "Optimizer/AdadeltaOptimizer.h"
//#include "Optimizer/AdamaxOptimizer.h"
//#include "Optimizer/RPropOptimizer.h"
//#include "Optimizer/RMSPropOptimizer.h"
//#include "Optimizer/AdamOptimizer.h"
//#include "Optimizer/NadamOptimizer.h"
//#include "Optimizer/AMSGradOptimizer.h"
//#include "Optimizer/FtrlOptimizer.h"
//#include "Optimizer/Optimizer.h"

#include "Visualizer/NetworkVisualizer.h"

//#include "ActivationFunction/Convolution2DFunction.h"
//#include "ActivationFunction/MaxPooling2DFunction.h"
//#include "ActivationFunction/FlattenFunction.h"
#include "ActivationFunction/ReLUFunction.h"
#include "ActivationFunction/SigmoidFunction.h"
//#include "ActivationFunction/SoftmaxFunction.h"

#include "Test.h"

//#define ALL
//#define FIVE
//#define FOUR
//#define THREE
//#define TWO
#define ONE
//#define SIGNAL
//#define IRIS
#define MNIST

#define VERBOSITY 0

const int PRINT = 100;
const double MIN_ERROR = 0.001f;
const int MAX_EPOCHS = 100000;
const double CONVERGENCE_W = 0.001;
const double CONVERGENCE_E = 0.00000001;

std::string getCurrentFolder();