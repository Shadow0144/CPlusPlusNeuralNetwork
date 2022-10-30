#pragma once

int cv_test();

void test_signal(int layers); 
void test_signal_reshape();
void test_signal_maxout();
void test_signal_crelu();

void test_iris(int layers);

void test_binary_mnist(); // MNIST with two classes
void test_mnist();

void test_catdog();