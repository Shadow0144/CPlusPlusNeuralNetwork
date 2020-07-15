#include "NetworkVisualizer.h"

#include <Eigen/Core>
#include <iostream>

#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/formhelper.h>

using namespace nanogui;

NetworkVisualizer::NetworkVisualizer()
{
    nanogui::init();
    Screen* screen = nullptr;
    screen = new Screen(Vector2i(500, 700), "NanoGUI test");

    bool bvar = true;
    int ivar = 12345678;
    double dvar = 3.1415926;
    float fvar = (float)dvar;
    std::string strval = "A string";
    Color colval(0.5f, 0.5f, 0.7f, 1.f);

    bool enabled = true;
    FormHelper* gui = new FormHelper(screen);
    ref<Window> window = gui->add_window(Vector2i(10, 10), "Form helper example");
    gui->add_group("Basic types");
    gui->add_variable("bool", bvar);
    gui->add_variable("string", strval);

    gui->add_group("Validating fields");
    gui->add_variable("int", ivar)->set_spinnable(true);
    gui->add_variable("float", fvar);
    gui->add_variable("double", dvar)->set_spinnable(true);

    gui->add_group("Complex types");
    gui->add_variable("Color", colval)
        ->set_final_callback([](const Color& c) {
        std::cout << "ColorPicker Final Callback: ["
            << c.r() << ", "
            << c.g() << ", "
            << c.b() << ", "
            << c.w() << "]" << std::endl;
            });

    gui->add_group("Other widgets");
    gui->add_button("A button", []() { std::cout << "Button pressed." << std::endl; });

    screen->set_visible(true);
    screen->perform_layout();
    window->center();

    nanogui::mainloop();
}

NetworkVisualizer::~NetworkVisualizer()
{

}