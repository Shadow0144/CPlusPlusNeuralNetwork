#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma warning(pop)

xt::xarray<double> test(xt::xarray<double> input);
void print_dims(xt::xarray<double> xarray);

cv::Mat convertToMat(xt::xarray<double> xtensor);
cv::Mat convertChannelToMat(xt::xarray<double> xtensor, int num = 0, int channel = 0, bool printDims = false);
cv::Mat convertKernelToMat3(xt::xarray<double> xtensor, int filter = 0, int numChannels = 1);
cv::Mat convertWeightsToMat3(xt::xarray<double> xtensor, int filter = 0, int numChannels = 1, int kernel = 0);
cv::Mat convertChannelsToMat3(xt::xarray<double> xtensor, int num = 0, int startChannel = 0, int numChannels = 1, bool printDims = false);