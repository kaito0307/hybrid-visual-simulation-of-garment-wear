#ifndef WEAR_HPP
#define WEAR_HPP

#include "mesh.hpp"

class Simulation;

void wear_step (Simulation &sim);
void clear_node_wear(Simulation &sim);
void empty_node_wear(Simulation &sim);


struct FaceWearCollection {
    double total_area = 0.;
    double sum_strain_energy = 0.;
    double sum_acc_strain_energy = 0.;

    ~FaceWearCollection() {
        if (sum_strain_energy > 1e-6) {
            assert (false);
        }
    }
};

struct NodeWearCollection {
    double total_mass = 0.;
    double sum_f_power[NodeWear::NumWearTypes] = {0., 0.};
    double sum_f_work[NodeWear::NumWearTypes] = {0., 0.};

    ~NodeWearCollection() {
        for (int i = 0; i < NodeWear::NumWearTypes; i++) {
            if (sum_f_power[i] > 1e-6) {
                assert (false);
            }
        }
    }
};

std::unique_ptr<FaceWearCollection> collect_face_wear(const std::vector<Face*> &faces);
std::unique_ptr<NodeWearCollection> collect_node_wear(const std::vector<Node*> &nodes);
void distribute_face_wear(const std::unique_ptr<FaceWearCollection> &collection, const std::vector<Face*> &faces);
void distribute_node_wear(const std::unique_ptr<NodeWearCollection> &collection, const std::vector<Node*> &nodes);
void transfer_node_wear(Node* o, Node *i, double d);

#endif
