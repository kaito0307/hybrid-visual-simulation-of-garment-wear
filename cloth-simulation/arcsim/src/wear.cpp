#include "wear.hpp"
#include "mesh.hpp"
#include "physics.hpp"

void clear_node_wear(Simulation &sim) {
    for (Cloth &cloth: sim.cloths) {
        for (Node *node: cloth.mesh.nodes) {
            if (!node->wear) continue;
            for (int i = 0; i < NodeWear::NumWearTypes; i++) {
                node->wear->f_power[i] = 0.0;
                node->wear->N[i] = Vec3(0.0);
            }
        }
    }
}

void empty_node_wear(Simulation &sim) {
    for (Cloth &cloth: sim.cloths) {
        for (Node *node: cloth.mesh.nodes) {
            if (!node->wear) continue;
            for (int i = 0; i < NodeWear::NumWearTypes; i++) {
                node->wear->f_work[i] = 0.0;
            }
        }
    }
}


void wear_step(Simulation &sim) {
    for (Cloth &cloth: sim.cloths) {
        double min_energy = std::numeric_limits<double>::max(),
            max_energy = -std::numeric_limits<double>::min();
        for (Face *face: cloth.mesh.faces) {
            if (face->wear == nullptr) {
                face->wear = std::make_unique<FaceWear>();
                face->wear->face = face;
            }

            face->wear->cur_strain_energy = stretching_energy<WS>(face);
            face->wear->acc_strain_energy += face->wear->cur_strain_energy * sim.step_time;
            min_energy = std::min(min_energy, face->wear->cur_strain_energy);
            max_energy = std::max(max_energy, face->wear->cur_strain_energy);
        }

        for (Node *node: cloth.mesh.nodes) {
            if (node->wear == nullptr) {
                node->wear = std::make_unique<NodeWear>();
                node->wear->node = node;
            }

            for (int i = 0; i < NodeWear::NumWearTypes; i++) {
                node->wear->f_work[i] += node->wear->f_power[i] * sim.step_time;
            }
        }
        printf("[Wear] min energy = %f, max energy = %f\n",
               min_energy, max_energy);
    }
}


std::unique_ptr<FaceWearCollection> collect_face_wear(const std::vector<Face*> &faces) {
    auto c = std::make_unique<FaceWearCollection>();
    c->sum_strain_energy = 0.0;
    c->sum_acc_strain_energy = 0.0;
    c->total_area = 0.0;

    for (Face *face: faces) {
        if (face->wear == nullptr) {
            face->wear = std::make_unique<FaceWear>();
            face->wear->face = face;
        }
        c->sum_strain_energy += face->wear->cur_strain_energy;
        c->sum_acc_strain_energy += face->wear->acc_strain_energy;
        c->total_area += face->a;

        face->wear->acc_strain_energy = 0;
    }

    if (c -> total_area == 0.0) {
        c->sum_strain_energy = 0.0;
    }
    return c;
}


std::unique_ptr<NodeWearCollection> collect_node_wear(const std::vector<Node*> &nodes) {
    auto c = std::make_unique<NodeWearCollection>();
    for (int i = 0; i < NodeWear::NumWearTypes; i++) {
        c->sum_f_power[i] = 0.0;
        c->sum_f_work[i] = 0.0;
    }
    c->total_mass = 0.0;

    if (nodes.size() == 0) {
        return c; // Return empty collection if no nodes
    }

    for (Node *node: nodes) {
        if (node->wear == nullptr) {
            node->wear = std::make_unique<NodeWear>();
            node->wear->node = node;
        }

        for (int i = 0; i < NodeWear::NumWearTypes; i++) {
            c->sum_f_power[i] += node->wear->f_power[i];
            c->sum_f_work[i] += node->wear->f_work[i];
            node->wear->f_power[i] = 0.0; // Reset power for next step
        }
        c->total_mass += node->m;
    }
    return c;
}


void distribute_face_wear(const std::unique_ptr<FaceWearCollection> &c, const std::vector<Face*> &faces) {
    double new_total_area = 0.0;
    for (Face *face: faces) {
        new_total_area += face->a;
    }
    for (Face *face: faces) {
        if (face->wear == nullptr) {
            face->wear = std::make_unique<FaceWear>();
            face->wear->face = face;
        }
        face->wear->cur_strain_energy = c->sum_strain_energy * (face->a / new_total_area);
        face->wear->acc_strain_energy = c->sum_acc_strain_energy * (face ->a / new_total_area);
    }

    c -> sum_strain_energy = 0.0;
    c -> sum_acc_strain_energy = 0.0;
}


void distribute_node_wear(const std::unique_ptr<NodeWearCollection> &c, const std::vector<Node*> &nodes) {
    if (c -> total_mass == 0.0) {
        return;
    }
    double new_total_mass = 0.;
    double distributed[NodeWear::NumWearTypes] = {0., 0.};
    for (Node *node: nodes) {
        new_total_mass += node->m;
    }
    for (Node *node: nodes) {
        if (node->wear == nullptr) {
            node->wear = std::make_unique<NodeWear>();
            node->wear->node = node;
        }

        for (int i = 0; i < NodeWear::NumWearTypes; i++) {
            node->wear->f_power[i] = c->sum_f_power[i] * (node->m / new_total_mass);
            node->wear->f_work[i] = c->sum_f_work[i] * (node->m / new_total_mass);
            distributed[i] += node -> wear -> f_work[i];
        }
    }

    for (int i = 0; i < NodeWear::NumWearTypes; i++) {
        c->sum_f_power[i] -= distributed[i];
    }
}

void transfer_node_wear(Node* o, Node *i, double d) {
    if (o -> wear == nullptr) {
        return; // No wear to transfer
    }
    if (i -> wear == nullptr) {
        i -> wear = std::make_unique<NodeWear>();
        i -> wear -> node = i;
    }

    for (int k = 0; k < NodeWear::NumWearTypes; k++) {
        i -> wear -> f_power[k] += o -> wear -> f_power[k] * d;
        o -> wear -> f_power[k] -= o -> wear -> f_power[k] * d;
        i -> wear -> f_work[k] += o -> wear -> f_work[k] * d;
        o -> wear -> f_work[k] -= o -> wear -> f_work[k] * d;
    }
}