#include "NetworkVisualizer.h"

#pragma warning(push, 0)
#include <iostream>
#include <Eigen/Core>
#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/formhelper.h>
#pragma warning(pop)

using namespace nanogui;

NetworkVisualizer::NetworkVisualizer()
{
    nanogui::init();
    Screen* screen = nullptr;
    screen = new Screen(Vector2i(500, 700), "Neural Network Visualizer");

    FormHelper* gui = new FormHelper(screen);
    ref<Window> window = gui->add_window(Vector2i(10, 10), "Form helper example");

    gui->add_group("Other widgets");
    gui->add_button("A button", []() { std::cout << "Button pressed." << std::endl; });

    screen->set_visible(true);
    screen->perform_layout();
    window->center();

    nanogui::mainloop();
}

NetworkVisualizer::~NetworkVisualizer()
{
    nanogui::shutdown();
}