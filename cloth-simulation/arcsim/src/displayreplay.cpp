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

#include "displayreplay.hpp"

#include "conf.hpp"
#include "display.hpp"
#include "io.hpp"
#include "iomeshseq.hpp"
#include "misc.hpp"
#include "opengl.hpp"
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <regex>
#include <GLFW/glfw3.h>
#include <boost/filesystem.hpp>
using namespace std;

#ifndef NO_OPENGL

extern string inprefix;
extern string outprefix;
static int frameskip;

static int max_frame;

static bool running = false;
static GLFWwindow* window = nullptr;

static void reload () {
	Annotation::list.clear();
    int fullframe = ::frame*::frameskip;

    if (frame >= max_frame) frame = max_frame - 1;
    if (frame < 0) frame = 0;

    sim.frame = ::frame;
    sim.time = fullframe * sim.frame_time;

    // if (!load_state(sim, stringf("%s/%05d",inprefix.c_str(), fullframe)) ||
    //     sim.cloth_meshes[0]->verts.empty()) {
    //     if (::frame == 0)
    //         exit(EXIT_FAILURE);
    //     if (!outprefix.empty())
    //         exit(EXIT_SUCCESS);
    //     ::frame = 0;
    //     reload();
    // }

    auto baselinefile = stringf("%s/%05d.npz",inprefix.c_str(), 27);
    if (!load_mesh_state(*sim.cloth_meshes[0],
        stringf("%s/%05d.npz",inprefix.c_str(), fullframe))) {
            if (::frame == 0)
                exit(EXIT_FAILURE);
            if (!outprefix.empty())
                exit(EXIT_SUCCESS);
            ::frame = 0;
            reload();
    }

    for (int o = 0; o < (int)sim.obstacles.size(); o++)
        sim.obstacles[o] -> get_mesh(sim.time);
}

static void idle () {
    if (!running)
        return;
    if (frame == max_frame - 1) {
        running = false;
        return;
    }
    fps.tick();
    if (!outprefix.empty()) {
        char filename[256];
        snprintf(filename, 256, "%s/%05d.png", outprefix.c_str(), ::frame );
        save_screenshot(filename);
    }
    ::frame += 1;
    reload();
    fps.tock();
    redisplay();
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (key == GLFW_KEY_ESCAPE) {
        exit(0);
    } else if (key == GLFW_KEY_SPACE) {
        running = !running;
    } else if (key == GLFW_KEY_LEFT) {
        int delta = (mods & GLFW_MOD_ALT) ? 100 : (mods & GLFW_MOD_SHIFT) ? 10 : 1;
        ::frame -= delta;
        reload();
        redisplay();
    } else if (key == GLFW_KEY_RIGHT) {
        int delta = (mods & GLFW_MOD_ALT) ? 100 : (mods & GLFW_MOD_SHIFT) ? 10 : 1;
        ::frame += delta;
        reload();
        redisplay();
    } else if (key == GLFW_KEY_A) {
        ::frame = 0;
        reload();
        redisplay();
    } else if (key == GLFW_KEY_Z) {
        ::frame = max_frame - 1;
        reload();
        redisplay();
    }
}

static void save_obstacle_transforms (const vector<Obstacle*> &obs, int frame,
                                      double time) {
    if (!outprefix.empty() && frame < 100000) {
        for (int o = 0; o < (int)obs.size(); o++) {
            Transformation trans = id_mat();
            if (obs[o] -> transform_spline)
                trans = get_dtrans(*obs[o] -> transform_spline, time).first;
            save_transformation(trans, stringf("%s/%05dobs%02d.txt",
                                               outprefix.c_str(), frame, o));
        }
    }
}

void generate_obj (const vector<string> &args) {
    if (args.size() < 1 || args.size() > 2) {
        cout << "Generates output meshes from saved simulation state." << endl;
        cout << "Arguments:" << endl;
        cout << "    <out-dir>: Directory containing simulation output files"
             << endl;
        exit(EXIT_FAILURE);
    }
    ::inprefix = args[0];
    ::outprefix = args[0];
    ::frameskip = 1;
    if (!::outprefix.empty())
        ensure_existing_directory(::outprefix);
    char config_backup_name[256];
    snprintf(config_backup_name, 256, "%s/%s", inprefix.c_str(), "conf.json");
    load_json(config_backup_name, sim);
    prepare(sim);
    
    for (;;) {
        reload();
        char filename[256];
        snprintf(filename, 256, "%s/%05d", inprefix.c_str(), ::frame);
        save_objs(sim.cloth_meshes, filename);
        save_obstacle_transforms(sim.obstacles, frame, sim.time);
        ::frame += 1;
    }
}

void check_max_frame() {
    namespace fs = std::filesystem;
    std::regex pattern(R"((\d+)\.npz)");  // matches digits followed by ".npz"
    std::vector<std::pair<int, fs::path>> numbered_files;

    for (const auto& entry : fs::directory_iterator(inprefix)) {
        if (entry.is_regular_file()) {
            std::smatch match;
            std::string filename = entry.path().filename().string();
            if (std::regex_match(filename, match, pattern)) {
                int number = std::stoi(match[1].str());
                numbered_files.emplace_back(number, entry.path());
            }
        }
    }

    std::sort(numbered_files.begin(), numbered_files.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });

    max_frame = numbered_files[numbered_files.size() - 1].first;
    std::printf("Set max_frame to %d\n", max_frame);
}

void render_loop() {
    string prefix = ::inprefix + "/render/";
    using namespace boost::filesystem;
    if (!exists(prefix))
        create_directory(prefix);

    for (int i = 0; i < max_frame; i += 1) {
        ::frame = i;
        reload();
        redisplay();
        if (!outprefix.empty()) {
            save_screenshot(stringf("%s/%05d.png", prefix.c_str(), frame + 1));
        }
    }
}

void display_replay (const vector<string> &args) {
    if (args.size() < 1 || args.size() > 3) {
        cout << "Replays the results of a simulation." << endl;
        cout << "Arguments:" << endl;
        cout << "    <out-dir>: Directory containing simulation output files or" << endl <<
                " \"quickpass\" to save the screenshot of the last frame" << endl;
        exit(EXIT_FAILURE);
    }
    ::inprefix = args[0];
    ::outprefix = args.size()>1 ? args[1] : "";
    bool quickpass = args.size() > 1 && args[1] == "quickpass";
    bool rendering = args.size() > 1 && args[1] == "render";
    ::frameskip = 1;
    if (!::outprefix.empty())
        ensure_existing_directory(::outprefix);
    char config_backup_name[256];
    snprintf(config_backup_name, 256, "%s/%s", inprefix.c_str(), "conf.json");
    load_json(config_backup_name, sim);
    check_max_frame();
    prepare(sim);
    reload();

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }
    if (quickpass) {
        int scale = 2;
        window = glfwCreateWindow(720 * scale, 1280 * scale, "Arcsim Replay", NULL, NULL);
        // We use a vertical window for quickpass to save screenshots
    }
    else if (rendering) {
        int scale = 2;
        window = glfwCreateWindow(1280 * scale, 720 * scale, "Arcsim Replay", NULL, NULL);
    }
    else {
        window = glfwCreateWindow(1280, 720, "Arcsim Replay", NULL, NULL);
    }
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
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

    if (quickpass) {
        ::Pane::panes[0].enabled = false; // material
        ::Pane::panes[1].enabled = false;
        ::Pane::panes[2].enabled = true; // world

        align_material_pane_rotation();

        frame = 1;
        reload();
        redisplay();
        save_screenshot(stringf("%s/%05d.png", inprefix.c_str(), frame + 1));

        ::Pane::panes[0].enabled = true; // material
        ::Pane::panes[1].enabled = false;
        ::Pane::panes[2].enabled = false; // world

        display_mode = 7; // Friction work ClothBody

        frame = max_frame - 1;
        reload();
        redisplay();
        save_screenshot(stringf("%s/%05d.png", inprefix.c_str(), frame + 1));
    }

    if (rendering) {
        display_mode = 5; // Friction power ClothBody
        align_material_pane_rotation();
        render_loop();
    }

    while (!glfwWindowShouldClose(window)) {
        if (quickpass || rendering) break;

        if (running) {
            idle();
        }
        redisplay();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

#else

void display_replay (const vector<string> &args) {opengl_fail();}
void generate_obj (const vector<string> &args) {opengl_fail();}

#endif // NO_OPENGL
