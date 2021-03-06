#include "NetworkVisualizer.h"
#include "NeuralNetwork.h"

#pragma warning(push, 0)
#include <Eigen/Dense>
#include <GL/gl3w.h>
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#pragma warning(pop)

#include "TanhFunction.h"
#include "SigmoidFunction.h"

NetworkVisualizer::NetworkVisualizer(NeuralNetwork* network)
{
    this->network = network;

    drag = ImVec2(0.0, 0.0);
    startDrag = false;
    origin = ImVec2(0.0, 0.0);
    scale = 1.0;

    inputsSet = false;
    predictedSet = false;
    targetsSet = false;

    function = NULL;
    displayFunctions = false;

    classifier = NULL;
    displayClasses = false;

    windowClosed = false;

    rendering = true;
    drawThread = thread(&NetworkVisualizer::draw, this);
}

NetworkVisualizer::~NetworkVisualizer()
{
    rendering = false;
    while (threadRunning);
    drawThread.join();

    if (function != NULL)
    {
        delete function;
    }
    else { }

    if (classifier != NULL)
    {
        delete classifier;
    }
    else { }
}

bool NetworkVisualizer::getWindowClosed()
{
    return windowClosed;
}

void NetworkVisualizer::setup()
{
    const int WINDOW_WIDTH = 1080;
    const int WINDOW_HEIGHT = 720;

    winSize = ImVec2(WINDOW_WIDTH, WINDOW_HEIGHT);

    // Setup SDL
    // (Some versions of SDL before <2.0.10 appears to have performance/stalling issues on a minority of Windows systems,
    // depending on whether SDL_INIT_GAMECONTROLLER is enabled or disabled.. updating to latest version of SDL is recommended!)
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
    }
    else { }

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    window = SDL_CreateWindow("Neural Network", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, winSize.x, winSize.y, window_flags);
    gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync

    bool err = gl3wInit() != 0;

    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
    }
    else { }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Setup the origin
    origin = ImVec2(winSize.x / 2.0, winSize.y / 2.0);
}

void test_draw(ImDrawList* draw_list)
{
    // Primitives 
    ImGui::Text("Primitives");
    static float sz = 36.0f;
    static float thickness = 4.0f;
    static ImVec4 col = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    ImGui::DragFloat("Size", &sz, 0.2f, 2.0f, 72.0f, "%.0f");
    ImGui::DragFloat("Thickness", &thickness, 0.05f, 1.0f, 8.0f, "%.02f");
    ImGui::ColorEdit4("Color", &col.x);
    {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImU32 col32 = ImColor(col);
        float x = p.x + 4.0f, y = p.y + 4.0f, spacing = 8.0f;

        float th = 1.0f;
        draw_list->AddCircle(ImVec2(x + sz * 0.5f, y + sz * 0.5f), sz * 0.5f, col32, 20, th); x += sz + spacing;    // Circle 
        draw_list->AddRect(ImVec2(x, y), ImVec2(x + sz, y + sz), col32, 0.0f, ImDrawCornerFlags_All, th); x += sz + spacing;
        draw_list->AddLine(ImVec2(x, y), ImVec2(x + sz, y + sz), col32, th); x += sz + spacing;               // Diagonal line 
        x = p.x + 4;
        y += sz + spacing;

        //draw_list->AddBezierCurve(ImVec2(x, y), ImVec2(x + sz * 1.0f, y + sz * 0.0f), ImVec2(x + sz * 0.0f, y + sz * 1.0f), ImVec2(x + sz * 1.0f, y + sz * 1.0f), col32, th);

        draw_list->AddCircleFilled(ImVec2(x + sz * 0.5f, y + sz * 0.5f), sz * 0.5f, col32, 32); x += sz + spacing;      // Circle 
        draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + sz, y + sz), col32); x += sz + spacing;
        draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + sz, y + sz), col32, 10.0f, ImDrawCornerFlags_TopLeft | ImDrawCornerFlags_BotRight); x += sz + spacing;
        x = p.x + 4;
        y += sz + spacing;
        y += sz + spacing;
        y += sz + spacing;

        /*TanhFunction thf(1);
        thf.draw(draw_list, ImVec2(x, y), sz);
        y += sz + spacing;
        y += sz + spacing;
        y += sz + spacing;

        SigmoidFunction sgf(1);
        sgf.draw(draw_list, ImVec2(x, y), sz);*/
    }
}

void NetworkVisualizer::addClassificationVisualization(int rows, int cols, ImColor* classColors)
{
    displayClasses = false;
    if (displayClasses)
    {
        delete classifier;
    }
    else { }

    classifier = new ClassifierVisualizer(this, rows, cols, classColors);
    displayClasses = true;
}

void NetworkVisualizer::addFunctionVisualization()
{
    displayFunctions = false;
    if (displayFunctions)
    {
        delete function;
    }
    else { }

    function = new FunctionVisualizer(this, network);
    displayFunctions = true;
}

ImVec2 NetworkVisualizer::getWindowSize()
{
    return winSize;
}

void NetworkVisualizer::setTargets(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
{
    resultsMutex.lock();
    this->inputs = xt::xarray<double>(inputs);
    this->targets = xt::xarray<double>(targets);
    inputsSet = true;
    targetsSet = true;
    resultsMutex.unlock();
}

void NetworkVisualizer::setPredicted(const xt::xarray<double>& predicted)
{
    resultsMutex.lock();
    this->predicted = xt::xarray<double>(predicted);
    predictedSet = true;
    resultsMutex.unlock();
}

void NetworkVisualizer::draw()
{
    threadRunning = true;

    setup();

    while (rendering)
    {
        renderFrame();
        Sleep(1);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    threadRunning = false;
}

bool NetworkVisualizer::getThreadRunning()
{
    return threadRunning;
}

void NetworkVisualizer::renderFrame()
{
    const ImVec4 CLEAR_COLOR(0.45, 0.55, 0.60, 1.0);
    const ImVec4 VERY_LIGHT_GRAY(0.8, 0.8, 0.8, 1.0);

    const int HALF_BORDER = 10;

    // Poll and handle events (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
    // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if ((event.type == SDL_QUIT) ||
            (event.type == SDL_WINDOWEVENT 
                && event.window.event == SDL_WINDOWEVENT_CLOSE
                && event.window.windowID == SDL_GetWindowID(window)))
        {
            windowClosed = true;
        }
        else { }
    }

    if (ImGui::GetIO().MouseDown[0])
    {
        ImVec2 newDrag = ImGui::GetMouseDragDelta(0);
        ImVec2 deltaDrag;
        if (!startDrag)
        {
            deltaDrag = newDrag;
            startDrag = true;
        }
        else
        {
            deltaDrag = ImVec2(newDrag.x - drag.x, newDrag.y - drag.y);
        }
        drag = newDrag;
        origin.x += deltaDrag.x;
        origin.y += deltaDrag.y;
    }
    else 
    { 
        startDrag = false;
    }
    double deltaScale = (ImGui::GetIO().MouseWheel * SCALE_FACTOR);
    if (deltaScale != 0) // TODO: Improve
    {
        double previousScale = scale;
        scale = max(MIN_SCALE_FACTOR, min(MAX_SCALE_FACTOR, scale + deltaScale));
        if (scale != previousScale)
        {
            ImVec2 mp = ImGui::GetIO().MousePos;
            origin.x -= mp.x * deltaScale;
            origin.y -= mp.y * deltaScale;
        }
        else { }
    }
    else { }

    origin.x = max((-MAX_PAN * scale), min(origin.x, (10 * MAX_PAN * scale)));
    origin.y = max((-MAX_PAN * scale), min(origin.y, (10 * MAX_PAN * scale)));

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(window); 
    ImGui::NewFrame();
    ImGui::PushStyleColor(ImGuiCol_WindowBg, VERY_LIGHT_GRAY);

    ImGui::SetNextWindowPos(ImVec2(HALF_BORDER, HALF_BORDER));
    ImGui::SetNextWindowSize(ImVec2(winSize.x - (2 * HALF_BORDER), winSize.y - (2 * HALF_BORDER)));

    ImGui::Begin("Network", 0, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    if (inputsSet && targetsSet)
    {
        if (resultsMutex.try_lock_shared())
        {
            network->draw(draw_list, origin, scale, inputs, targets);

            if (displayFunctions)
            {
                function->draw(draw_list, inputs, targets);
            }
            else { }

            if (displayClasses && predictedSet)
            {
                classifier->draw(draw_list, predicted, targets);
            }
            else { }
            resultsMutex.unlock_shared();
        }
        else { }
    }
    else { }

    ImGui::End();
    ImGui::PopStyleColor();

    // Rendering
    ImGui::Render();
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    glClearColor(CLEAR_COLOR.x, CLEAR_COLOR.y, CLEAR_COLOR.z, CLEAR_COLOR.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window);
}