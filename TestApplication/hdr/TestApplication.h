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
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma warning(pop)

#include "NeuralNetwork.h"

//#include "MeanSquareErrorLossFunction.h"
//#include "CrossEntropyErrorLossFunction.h"

//#include "SGDOptimizer.h"
//#include "AdagradOptimizer.h"
//#include "AdadeltaOptimizer.h"
//#include "AdamaxOptimizer.h"
//#include "RPropOptimizer.h"
//#include "RMSPropOptimizer.h"
//#include "AdamOptimizer.h"
//#include "NadamOptimizer.h"
//#include "AMSGradOptimizer.h"
//#include "FtrlOptimizer.h"
//#include "Optimizer.h"

#include "NetworkVisualizer.h"

//#include "Convolution2DFunction.h"
//#include "MaxPooling2DFunction.h"
//#include "FlattenFunction.h"
#include "ReLUFunction.h"
#include "SigmoidFunction.h"
//#include "SoftmaxFunction.h"

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