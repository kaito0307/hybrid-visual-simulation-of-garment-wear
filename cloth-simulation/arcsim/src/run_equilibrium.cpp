#include "run_equilibrium.hpp"
#include "display.hpp"
#include "io.hpp"
#include "conf.hpp"
#include <iostream>
#include <vector>
#include <boost/filesystem/operations.hpp>

extern Simulation sim;

static int step_cnt;

#include "runphysics.hpp"

namespace equilibrium {
    void run_equilibrium(const std::vector<std::string> &args) {
        using namespace std;
        if (args.size() != 1 && args.size() != 2) {
            cout << "Runs the simulation in batch mode." << endl;
            cout << "Arguments:" << endl;
            cout << "    <scene-file>: JSON file describing the simulation setup"
                 << endl;
            cout << "    <out-dir> (optional): Directory to save output in" << endl;
            exit(EXIT_FAILURE);
        }
        string json_file = args[0];
        string outprefix = args.size()>1 ? args[1] : "";
        if (!outprefix.empty())
            ensure_existing_directory(outprefix);
        init_equilibrium(json_file, outprefix, false);
        init_relax();
        if (!outprefix.empty())
            save(sim, 0);
        equilibrium_loop();
    }

    void equilibrium_loop() {
        while (true) {
            sim_step(true);
            sim_step(false);
        }
    }

    void init_relax() {

    }

    void load_initial_pos(float time, Simulation &sim, bool reset_garments) {
        if (reset_garments)
            for (auto cloth: sim.cloths) {
                auto mesh = cloth.mesh;
                cloth.initial_positions -> get_positions(time, mesh.nodes);
                compute_ws_data(mesh);
            }

        for (int o = 0; o < (int)sim.obstacles.size(); o++)
            sim.obstacles[o] -> get_mesh(sim.time);

        step_cnt = 0;
    }

    void init_equilibrium (const std::string &json_file, std::string outprefix,
                       bool is_reloading) {
        load_json(json_file, sim);
        // ::outprefix = outprefix;
        if (!outprefix.empty()) {
            // ::timingfile.open(stringf("%s/timing", outprefix.c_str()).c_str(),
            //                   is_reloading ? ios::out|ios::app : ios::out);
            // Make a copy of the config file for future use
            boost::filesystem::copy_file(json_file.c_str(), stringf("%s/conf.json",outprefix.c_str()));
            // And copy over all the obstacles
            std::vector<Mesh*> base_meshes(sim.obstacles.size());
            for (int o = 0; o < (int)sim.obstacles.size(); o++)
                base_meshes[o] = &sim.obstacles[o] -> base_mesh;
            save_objs(base_meshes, stringf("%s/obs", outprefix.c_str()));
        }
        prepare(sim);
        save(sim, 0);
    }


    void sim_step(bool increase_time) {
        fps.tick();
        if (increase_time) {
            // This allows time to move forward
            // We don't alawys want time to move because we would iterate multiple times as we
            // solve for equilibrium
            sim.time += sim.step_time;
            sim.step++;
            load_initial_pos(sim.time, sim, false);
        }
        advance_equilibrium(sim);
        step_cnt++;
        if ((sim.step % sim.frame_steps == 0) && (sim.step % sim.save_every) == 0 && increase_time) {
            sim.frame++;
        }
        if ((sim.step % sim.frame_steps == 0) && (sim.step % sim.save_every) == 0) {
            printf("[Sim step] Trying to save\n");
            save(sim, sim.frame);
        }
        fps.tock();
        printf("[run equilibrium]:  FPS %.2f, Step %d\n",1.0 / fps.last, step_cnt);
        if (sim.time >= sim.end_time || sim.frame >= sim.end_frame)
            exit(EXIT_SUCCESS);
    }
}
