#ifndef ARCSIM_RUN_EQUILIBRIUM_HPP
#define ARCSIM_RUN_EQUILIBRIUM_HPP

#include "simulation.hpp"

namespace equilibrium {
    void init_equilibrium (const std::string &json_file, std::string outprefix, bool is_reloading);
    void sim_step(bool increase_time = true);
    void run_equilibrium (const std::vector<std::string> &args);
    void init_relax();
    void load_initial_pos(float time, Simulation &sim, bool reset_garments=true);
    void equilibrium_loop();
}


#endif //ARCSIM_RUN_EQUILIBRIUM_HPP