#pragma once
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

void loadCatDogDataset(xt::xarray<double>& features, xt::xarray<double>& labels, bool shuffle = true)
{
	std::string catLocation = "C:/Users/Corbi/Pictures/CNN/cat_archive/";
	std::string dogLocation = "C:/Users/Corbi/Pictures/CNN/dog_archive/";

	const int MAX_CATS = 1;// 10;// 0;
	const int MAX_DOGS = 1;// 10;// 0;

	// The network relies on the images being this size
	size_t maxWidth = 1024;
	size_t maxHeight = 1024;
	size_t channels = 3;

	size_t N = 0;
	cv::Mat image;
	vector<cv::String> fn;

	cv::glob(catLocation + "*.jpg", fn, true);
	vector<cv::Mat> catImages;
	size_t catCount = min((int)fn.size(), MAX_CATS);
	N += catCount;
	for (int i = 0; i < catCount; i++)
	{
		image = cv::imread(fn[i], cv::IMREAD_COLOR);
		cv::Mat fImage;
		image.convertTo(fImage, CV_32FC3);
		catImages.push_back(fImage);
		maxWidth = max((int)maxWidth, image.cols);
		maxHeight = max((int)maxHeight, image.rows);
	}

	cv::glob(dogLocation + "*.jpg", fn, true);
	vector<cv::Mat> dogImages;
	size_t dogCount = min((int)fn.size(), MAX_DOGS);
	N += dogCount;
	for (int i = 0; i < dogCount; i++)
	{
		image = cv::imread(fn[i], cv::IMREAD_COLOR);
		cv::Mat fImage;
		image.convertTo(fImage, CV_32FC3);
		dogImages.push_back(fImage);
		maxWidth = max((int)maxWidth, image.cols);
		maxHeight = max((int)maxHeight, image.rows);
	}

	auto featureShape = xt::svector<size_t>({ N, maxHeight, maxWidth, channels });
	features = xt::zeros<double>(featureShape);

	auto labelShape = xt::svector<size_t>({ N, 2 });
	labels = xt::zeros<double>(labelShape);

	int index = 0;
	for (int n = 0; n < catCount; n++)
	{
		cv::Mat catImage = catImages.at(n);
		int H = catImage.rows;
		int W = catImage.cols;
		for (int h = 0; h < H; h++)
		{
			for (int w = 0; w < W; w++)
			{
				for (int c = 0; c < channels; c++)
				{
					features(n + index, h, w, c) = catImage.at<cv::Vec3f>(h, w)[c];
				}
			}
		}
		labels(n + index, 0) = 1; // Cat
	}
	index += catCount;

	for (int n = 0; n < dogCount; n++)
	{
		int H = dogImages.at(n).rows;
		int W = dogImages.at(n).cols;
		for (int h = 0; h < H; h++)
		{
			for (int w = 0; w < W; w++)
			{
				for (int c = 0; c < channels; c++)
				{
					features(n + index, h, w, c) = dogImages.at(n).at<cv::Vec3f>(h, w)[c];
				}
			}
		}
		labels(n + index, 1) = 1; // Dog
	}
	//index += dogCount;

	// Shuffle
	if (shuffle)
	{
		xt::xstrided_slice_vector svI({ 0, xt::ellipsis() });
		xt::xstrided_slice_vector svJ({ 0, xt::ellipsis() });
		for (int i = (N - 1); i > 0; i--)
		{
			int j = rand() % i;
			svI[0] = i;
			svJ[0] = j;
			auto x = xt::xarray<double>(xt::strided_view(features, svI));
			xt::strided_view(features, svI) = xt::strided_view(features, svJ);
			xt::strided_view(features, svJ) = x;
			auto y = xt::xarray<double>(xt::strided_view(labels, svI));
			xt::strided_view(labels, svI) = xt::strided_view(labels, svJ);
			xt::strided_view(labels, svJ) = y;
		}
	}
	else { }
}