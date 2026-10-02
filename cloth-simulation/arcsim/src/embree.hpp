#ifndef EMBREE_HPP
#define EMBREE_HPP

#include "collisionutil.hpp"
#include "mesh.hpp"
#include <embree4/rtcore.h>
#include <tbb/tbb.h>


void for_overlapping_faces_embree(
    const std::vector<Mesh*>& meshes,
    const std::vector<Mesh*>& obs_meshes,
    double thickness,
    BVHCallback callback,
    bool parallel=true,
    bool only_obs=false);


// Embree geometry/scene wrapper for update
struct EmbreeAccel {
    RTCScene scene;
    RTCGeometry geom;
    float* vertex_buffer;
    size_t num_vertices;
    EmbreeAccel() : scene(nullptr), geom(nullptr), vertex_buffer(nullptr), num_vertices(0) {}
};

EmbreeAccel create_embree_accel_from_mesh(const Mesh& mesh, RTCDevice device);
void update_embree_accel(const Mesh& mesh, EmbreeAccel& acc);
void destroy_embree_accel(EmbreeAccel& acc);


#endif