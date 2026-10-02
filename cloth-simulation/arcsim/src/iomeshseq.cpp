#include <fstream>
#include <vector>
#include <iostream>
#include <Eigen/Core>

#include "iomeshseq.hpp"

#include <boost/filesystem.hpp>

#include "cnpy.h"
#include "mesh.hpp"

constexpr size_t n_damage = 11; // Number of damage-related properties to save
constexpr size_t n_stretching = 2; // Number of stretching-related properties to save

// This is no longer used, but kept for reference
bool load_frames_binary(const std::string& filename, std::vector<Eigen::MatrixX3f>& frames) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    uint32_t num_frames = 0, num_vertices = 0;
    in.read(reinterpret_cast<char*>(&num_frames), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&num_vertices), sizeof(uint32_t));

    frames.resize(num_frames, Eigen::MatrixX3f(num_vertices, 3));

    for (uint32_t f = 0; f < num_frames; ++f) {
        for (uint32_t v = 0; v < num_vertices; ++v) {
            float x, y, z;
            in.read(reinterpret_cast<char*>(&x), sizeof(float));
            in.read(reinterpret_cast<char*>(&y), sizeof(float));
            in.read(reinterpret_cast<char*>(&z), sizeof(float));
            frames[f].row(v) = Eigen::Vector3f(x, y, z);
        }
    }
    return true;
}


bool load_frames(const std::string &filename, std::vector<Eigen::MatrixX3f>& frames) {
    auto load = cnpy::npy_load(filename);
    if (load.word_size != sizeof(float) || load.fortran_order) {
        std::cerr << "Error: Invalid file format for " << filename << std::endl;
        return false;
    }

    float* data = load.data<float>();
    size_t num_frames = load.shape[0];
    size_t num_vertices = load.shape[1];
    frames.resize(num_frames, Eigen::MatrixX3f(num_vertices, 3));

    for (size_t f = 0; f < num_frames; ++f) {
        float *fd = data + f * num_vertices * 3;
        for (size_t v = 0; v < num_vertices; ++v) {
            frames[f].row(v) = Eigen::Vector3f(fd[v * 3],
                                               fd[v * 3 + 1],
                                               fd[v * 3 + 2]);
        }
    }

    return true;
}


void save_mesh_state(const Mesh& mesh, const std::string& filename) {
    size_t N = mesh.nodes.size();

    float *buffer = new float[N * 3];
    auto p = reinterpret_cast<float(*)[3]>(buffer);

    // Save positions
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        const Node* node = mesh.nodes[i];
        p[i][0] = node->x[0];
        p[i][1] = node->x[1];
        p[i][2] = node->x[2];
    }
    cnpy::npz_save(filename, "positions", buffer, {N, 3},  "w");
    delete[] buffer;

    // Save damage
    buffer = new float[N * n_damage];
    auto d = reinterpret_cast<float(*)[n_damage]>(buffer);
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        const Node* node = mesh.nodes[i];
        d[i][0] = node->a;
        d[i][1] = node->wear ? node->wear->f_work[0] : 0;
        d[i][2] = node->wear ? node->wear->f_power[0] : 0;
        d[i][3] = node->wear ? node->wear->f_work[1] : 0;
        d[i][4] = node->wear ? node->wear->f_power[1] : 0;
        for (int j = 0; j < 3; j++)
            d[i][5 + j] = node->wear ? node->wear->N[0][j] : 0;
        for (int j = 0; j < 3; j++)
            d[i][8 + j] = node->wear ? node->wear->N[1][j] : 0;
    }

    cnpy::npz_save(filename, "damage", buffer, {N, n_damage}, "a");
    delete[] buffer;

    N = mesh.faces.size();

    buffer = new float[N * n_stretching];
    auto s = reinterpret_cast<float(*)[n_stretching]>(buffer);
    // Save stretching data
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        const Face *face = mesh.faces[i];
        s[i][0] = face->wear ? face->wear->cur_strain_energy : 0;
        s[i][1] = face->wear ? face->wear->acc_strain_energy : 0;
    }

    cnpy::npz_save(filename, "stretching", buffer, {N, n_stretching}, "a");
    delete[] buffer;
}


template<class T>
T non_neg(T x) {
    return x < 0 ? 0 : x;
}


bool load_mesh_state(Mesh &mesh, const std::string& filename, const std::string& baseline_filename) {
    size_t N = mesh.nodes.size();

    auto positions = cnpy::npz_load(filename, "positions");
    if (positions.word_size != sizeof(float) || positions.fortran_order) {
        std::cerr << "Error: Invalid file format for " << filename << std::endl;
        return false;
    }

    float* pos_data = positions.data<float>();
    auto p = reinterpret_cast<float(*)[3]>(pos_data);

    for (int i = 0; i < N; i++) {
        mesh.nodes[i]->x[0] = p[i][0];
        mesh.nodes[i]->x[1] = p[i][1];
        mesh.nodes[i]->x[2] = p[i][2];
    }

    compute_ws_data(mesh);

    auto damage = cnpy::npz_load(filename, "damage");
    if (damage.word_size != sizeof(float) || damage.fortran_order) {
        std::cerr << "Error: Invalid file format for " << filename << std::endl;
        return false;
    }

    float* damage_data = damage.data<float>();
    auto d = reinterpret_cast<float(*)[damage.shape[1]]>(damage_data);

    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        if (mesh.nodes[i]->wear == nullptr) {
            mesh.nodes[i]->wear = std::make_unique<NodeWear>();
        }

        mesh.nodes[i]->wear->f_work[0] = d[i][1];
        mesh.nodes[i]->wear->f_power[0] = d[i][2];
        mesh.nodes[i]->wear->f_work[1] = d[i][3];
        mesh.nodes[i]->wear->f_power[1] = d[i][4];

        if (damage.shape[1] >= 11) {
            for (int j = 0; j < 3; j++)
                mesh.nodes[i]->wear->N[0][j] = d[i][5 + j];
            for (int j = 0; j < 3; j++)
                mesh.nodes[i]->wear->N[1][j] = d[i][8 + j];
        }
    }

    // Check if the file of baseline_filename exists
    if (!baseline_filename.empty()) {
        auto baseline_damage = cnpy::npz_load(baseline_filename, "damage");
        if (baseline_damage.word_size != sizeof(float) || baseline_damage.fortran_order) {
            std::cerr << "Error: Invalid file format for " << baseline_filename << std::endl;
            return true;
        }

        float* baseline_data = baseline_damage.data<float>();
        auto b = reinterpret_cast<float(*)[baseline_damage.shape[1]]>(baseline_data);

        if (baseline_damage.shape[0] != mesh.nodes.size() || baseline_damage.shape[1] < 5) {
            std::cerr << "Error: Baseline damage file has incompatible shape." << std::endl;
            return true;
        }

        for (size_t i = 0; i < mesh.nodes.size(); ++i) {
            if (mesh.nodes[i]->wear == nullptr) {
                mesh.nodes[i]->wear = std::make_unique<NodeWear>();
            }
            // Overwrite the wear data with the baseline values
            mesh.nodes[i]->wear->f_work[0] -= b[i][1];
            mesh.nodes[i]->wear->f_work[0] = non_neg(mesh.nodes[i]->wear->f_work[0]);
            // mesh.nodes[i]->wear->f_power[0] = b[i][2];
            mesh.nodes[i]->wear->f_work[1] -= b[i][3];
            mesh.nodes[i]->wear->f_work[1] = non_neg(mesh.nodes[i]->wear->f_work[1]);
            // mesh.nodes[i]->wear->f_power[1] = b[i][4];
        }
    }

    try {
        auto stretching = cnpy::npz_load(filename, "stretching");
        float *stretching_data = stretching.data<float>();
        auto c = reinterpret_cast<float(*)[n_stretching]>(stretching_data);
        for (size_t i = 0; i < mesh.faces.size(); ++i) {
            if (mesh.faces[i]->wear == nullptr) {
                mesh.faces[i]->wear = std::make_unique<FaceWear>();
            }
            mesh.faces[i]->wear->cur_strain_energy = c[i][0];
            mesh.faces[i]->wear->acc_strain_energy = c[i][1];
        }
    } catch (const std::exception& e) {
        for (size_t i = 0; i < mesh.faces.size(); ++i) {
            if (mesh.faces[i]->wear == nullptr) {
                continue;
            }
            mesh.faces[i]->wear->cur_strain_energy = 0;
            mesh.faces[i]->wear->acc_strain_energy = 0;
        }
    }

    return true;
}
