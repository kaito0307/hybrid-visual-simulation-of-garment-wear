#ifndef NO_OPENGL

#include "display_equilibrium.h"

#include "geometry.hpp"
#include "io.hpp"
#include "display.hpp"
#include "opengl.hpp"
#include "misc.hpp"
#include "run_equilibrium.hpp"
#include "simulation.hpp"

using namespace std;

extern string outprefix;
extern fstream timingfile;

static bool running = false;
static bool increase_time = false;
static GLFWwindow* window = nullptr;
static int step_cnt;


static void step_and_redisplay() {
    if (!::running)
        return;
    step_cnt++;
    advance_equilibrium(sim);
    printf("[Display Equilibrium] Step %d\n", step_cnt);
    redisplay();
}


static void initialize_frame() {
    equilibrium::load_initial_pos(sim.time, sim);
    redisplay();
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_SPACE) {
        ::running = !::running;
    } else if (key == GLFW_KEY_S) {
        ::running = !::running;
        step_and_redisplay();
        ::running = !::running;
    } else if (key == GLFW_KEY_RIGHT) {
        sim.time += sim.frame_time;
        sim.frame += 1;
        initialize_frame();
    } else if (key == GLFW_KEY_LEFT) {
        sim.time -= sim.frame_time;
        sim.frame -= 1;
        if (sim.time < 0) sim.time = 0;
        initialize_frame();
    } else if (key == GLFW_KEY_T) {
        increase_time = !increase_time;
        printf("[Display equilibrium] increase time set to %d\n", increase_time);
    }
}


void display_equilibrium (const vector<string> &args) {
    if (args.size() != 1 && args.size() != 2) {
        cout << "Runs the simulation with an OpenGL display." << endl;
        cout << "Arguments:" << endl;
        cout << "    <scene-file>: JSON file describing the simulation setup" << endl;
        cout << "    <out-dir> (optional): Directory to save output in" << endl;
        exit(EXIT_FAILURE);
    }
    string json_file = args[0];
    outprefix = args.size()>1 ? args[1] : "";
    if (!outprefix.empty())
        ensure_existing_directory(outprefix);

    equilibrium::init_equilibrium(json_file, outprefix, false);
    equilibrium::init_relax();

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_SAMPLES, 4); // Enable 4x MSAA
    window = glfwCreateWindow(1024, 768, "Arcsim", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glEnable(GL_MULTISAMPLE); // Enable multisampling in OpenGL
    add_key_callback(key_callback);
    prepare_callbacks();

    ::Pane::panes[0] = Pane(true); // material
    ::Pane::panes[1] = Pane(false); // plastic
    ::Pane::panes[2] = Pane(true); // world

    ::Pane::panes[0].offset[1] -= 0.4;
    ::Pane::panes[0].scale = 0.4;
    ::Pane::panes[0].offset[0] -= 0.15;

    ::Pane::panes[2].offset[1] -= 0.3;
    ::Pane::panes[2].scale = 0.4;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        if (running) {
            equilibrium::sim_step(increase_time);
            if (increase_time) {
                for (int iter = 0; iter < 5; iter++)
                    equilibrium::sim_step(false);
            }
        }
        redisplay();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

#else // NO_OPENGL

#include "display_equilibrium.h"
#include <vector>
#include <string>
#include <iostream>
using namespace std;

void display_equilibrium (const vector<string> &args) {
    cerr << "display_equilibrium: built without OpenGL support" << endl;
}

#endif // NO_OPENGL