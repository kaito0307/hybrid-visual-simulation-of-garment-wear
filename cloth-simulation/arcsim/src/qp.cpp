#include "qp.hpp"
#include "collisionutil.hpp"
#include <Eigen/Dense>

#include "magic.hpp"


constexpr double obs_mass = 1e3;
static double get_mass (const Node *node) {
    return is_free(node) ? node->m : obs_mass;}


void projected_gauss_seidel(ImpactZone *zone, std::unordered_map<const Node*, Vec3> &xold) {
    double thickness = ::magic.projection_thickness;
    constexpr int max_iter = 10;
    int n_nodes = zone->nodes.size();
    int n_cons = zone->impacts.size();

    float inv_m = 0;
    for (int n = 0; n < (int)zone->nodes.size(); n++)
        inv_m += 1/get_mass(zone->nodes[n]);
    inv_m /= zone->nodes.size();

    std::unordered_map<int, int> node_idx2zone_idx;

    for (int i = 0; i < zone -> nodes.size(); i++)
        node_idx2zone_idx[zone -> nodes[i] -> uuid] = i;

    Eigen::VectorXf xx(n_nodes * 3);
    Eigen::VectorXf xxold(n_nodes * 3);

    Eigen::VectorXf con_grad(n_nodes * 3);
    Eigen::VectorXf obj_grad(n_nodes * 3);
    con_grad.setZero();
    obj_grad.setZero();

    Eigen::VectorXf nn(n_cons * 3);
    Eigen::VectorXf ww(n_cons * 4);

    // Initialize x and xold from current node positions
    for (int i = 0; i < n_nodes; i++) {
        xx(3*i + 0) = zone->nodes[i]->x[0];
        xx(3*i + 1) = zone->nodes[i]->x[1];
        xx(3*i + 2) = zone->nodes[i]->x[2];

        Vec3 old = xold[zone->nodes[i]];
        xxold(3*i + 0) = old[0];
        xxold(3*i + 1) = old[1];
        xxold(3*i + 2) = old[2];
    }

    for (int i = 0; i < n_cons; i++) {
        nn(3*i + 0) = zone->impacts[i].n[0];
        nn(3*i + 1) = zone->impacts[i].n[1];
        nn(3*i + 2) = zone->impacts[i].n[2];

        ww(4*i + 0) = zone->impacts[i].w[0];
        ww(4*i + 1) = zone->impacts[i].w[1];
        ww(4*i + 2) = zone->impacts[i].w[2];
        ww(4*i + 3) = zone->impacts[i].w[3];
    }

    // for (int j = 0; j < n_cons; j++) {
    //     const Impact &impact = zone->impacts[j];
    //     for (int n = 0; n < 4; n++) {
    //         if (node_idx2zone_idx.contains(impact.nodes[n] -> uuid)) {
    //             int i = node_idx2zone_idx[impact.nodes[n] -> uuid];
    //             con_grad.segment<3>(3*i) += impact.w[n] * nn.segment<3>(3*j);
    //         }
    //     }
    // }

    for (int iter = 0; iter < max_iter; iter++) {
        bool all_good = true;
        // printf("[PGS] Iteration %d of size %d\n", iter, n_cons);
        // --- Step 1: free motion update (minimize quadratic objective) ---
        for (int i = 0; i < n_nodes; i++) {
            double m = get_mass(zone->nodes[i]);
            Eigen::Vector3f dx = xx.segment<3>(3*i) - xxold.segment<3>(3*i);
            xx.segment<3>(3*i) -= inv_m * m * dx;  // or multiply by some inv_m if scaled differently
        }

        // --- Step 2: project violated constraints ---
        for (int j = 0; j < n_cons; j++) {
            Eigen::Vector4f w = ww.segment<4>(4*j);
            Eigen::Vector3f n = nn.segment<3>(3*j);
            const Impact &impact = zone->impacts[j];

            // Compute current constraint
            float c = -thickness;
            for (int k = 0; k < 4; k++) {
                int node_idx = node_idx2zone_idx[impact.nodes[k] -> uuid];
                c += w[k] * n.dot(xx.segment<3>(3*node_idx));
            }

            // If violated, project
            if (c < 0) {
                all_good = false;
                float wsum = w.dot(w);
                float lambda = -c / (n.dot(n) * (w.dot(w)));

                for (int k = 0; k < 4; k++) {
                    int node_idx = node_idx2zone_idx[impact.nodes[k] -> uuid];
                    xx.segment<3>(3*node_idx) += (w(k) / wsum) * lambda * n;
                }
            }
        }

        if (all_good) break;
    }

    // Write back final positions
    for (int i = 0; i < n_nodes; i++) {
        zone->nodes[i]->x = Vec3(xx(3*i), xx(3*i + 1), xx(3*i + 2));
    }
}
