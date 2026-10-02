#include "materialplot.hpp"

#include <fstream>
#include <iostream>
#include <json/json.h>
#include "io.hpp"
#include "conf.hpp"
#include "geometry.hpp"
#include "cnpy.h"


Mat3x3 material_model (const Material* mat, const Mat3x3& G) {
    if (mat->use_dde) {
        Vec4 k = stretching_stiffness(reduce_xy(G), mat->dde_stretching);
        Mat3x3 sigma (Vec3(k[0]*G(0,0)+k[1]*G(1,1), 0.5*k[3]*G(0,1), 0),
                      Vec3(0.5*k[3]*G(1,0), k[2]*G(1,1)+k[1]*G(0,0), 0),
                      Vec3(0, 0, 0));
        return sigma;
    } else {
        double A = mat->alt_stretching;
        Mat3x3 sigma = A*(1.0-mat->alt_poisson)*G + Mat3x3(A*mat->alt_poisson*trace(G));
        return sigma;
    }
}


double stretching_energy (const Material *mat, const Mat3x3 &F) {
    Mat3x3 G = (F.t()*F - Mat3x3(1)) * 0.5;
    Mat3x3 sigma = material_model(mat, G);

    return 0.5 * inner(sigma, G);
}


double get_energy(const Material *mat, double scale_factor, double rotational_angle, int stretching_axis) {
    // Rotate the material along z axis and stretch it in y direction
    // Note the rotation uses the inverse angle because the rotation is done in material space,
    // corresponding to an inverse rotation (on the rhs) in the deformation gradient.
    Mat3x3 F;
    Mat3x3 R;
    double a = -rotational_angle / 180 * M_PI; // convert to radians
    F(0, 0) = F(1, 1) = F(2, 2) = 1;
    F(stretching_axis, stretching_axis) *= scale_factor;

    R(0, 0) = cos(a);
    R(0, 1) = -sin(a);
    R(1, 0) = sin(a);
    R(1, 1) = cos(a);

    F = F * R;

    return stretching_energy(mat, F);
}

void materialplot(const std::vector<std::string> &args) {
    Material* material;
    if (args.size() != 3) {
        std::cout << "Usage: materialplot <material.json> <n_point> <out.npy>\n";
    }
    Json::Value json;
    Json::Reader reader;
    std::string filename = args[0];
    int n_point = std::atoi(args[1].c_str());
    std::string out_filename = args[2];
    std::ifstream file(filename);

    bool ok = reader.parse(file, json);


    if (!ok) {
        fprintf(stderr, "Error reading json file %s\n", filename.c_str());
    }

    parse(material, json);
    std::vector<double> results;

    double start = 0, end = 360, step = (end - start) / n_point;
    for (int a = 0; a < 2; a++) {
        for (int i = 0; i < n_point; i++) {
            double angle = start + i * step;
            double energy = get_energy(material, 1.2, angle, a);
            results.push_back(energy);
        }
    }

    std::vector shape = {2, static_cast<size_t>(n_point)};
    // Write the result to npy file
    cnpy::npy_save(out_filename, &results[0], shape, "w");
}