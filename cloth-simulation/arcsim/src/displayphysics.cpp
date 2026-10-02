/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "displayphysics.hpp"

#include "display.hpp"
#include "io.hpp"
#include "opengl.hpp"
#include "misc.hpp"
#include "runphysics.hpp"
#include "simulation.hpp"
#include "timer.hpp"
#include "util.hpp"

#include <GLFW/glfw3.h>
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "run_equilibrium.hpp"
using namespace std;

#ifndef NO_OPENGL

extern string outprefix;
extern fstream timingfile;

static bool running = false;
static GLFWwindow* window = nullptr;

static void step_and_redisplay() {
    if (!::running)
        return;
    sim_step();
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
    } else if (key == GLFW_KEY_E) {
        equilibrium::sim_step();
        redisplay();
    }
}

void display_physics (const vector<string> &args) {
    if (args.size() != 1 && args.size() != 2) {
        cout << "Runs the simulation with an OpenGL display." << endl;
        cout << "Arguments:" << endl;
        cout << "    <scene-file>: JSON file describing the simulation setup" << endl;
        cout << "    <out-dir> (optional): Directory to save output in" << endl;
        exit(EXIT_FAILURE);
    }
    string json_file = args[0];
    string outprefix = args.size()>1 ? args[1] : "";
    if (!outprefix.empty())
        ensure_existing_directory(outprefix);

    init_physics(json_file, outprefix, false);
    init_relax();
    if (!outprefix.empty())
        save(sim, 0);

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
            sim_step();
        }
        redisplay();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

void display_resume (const vector<string> &args) {
    if (args.size() != 2) {
        cout << "Resumes an incomplete simulation." << endl;
        cout << "Arguments:" << endl;
        cout << "    <out-dir>: Directory containing simulation output files" << endl;
        cout << "    <resume-frame>: Frame number to resume from" << endl;
        exit(EXIT_FAILURE);
    }
    init_resume(args);
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
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        if (running) {
            sim_step();
        }
        redisplay();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

#else

void display_physics (const vector<string> &args) {opengl_fail();}

void display_resume (const vector<string> &args) {opengl_fail();}

#endif // NO_OPENGL
