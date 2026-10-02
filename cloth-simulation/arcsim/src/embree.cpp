#include "embree.hpp"
#include <vector>
#include <functional>
#include <limits>
#include <cstring>
#include <omp.h>
#include "collisionutil.hpp"
#include <embree4/rtcore.h>
#include <embree4/rtcore_common.h>


// Helper: Convert Mesh to Embree RTCScene (Embree 4)
// RTCScene create_embree_scene_from_mesh(const Mesh& mesh, RTCDevice device) {
//     RTCScene scene = rtcNewScene(device);
//     RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
//
//     // Prepare vertex buffer
//     std::vector<float> vertices;
//     vertices.reserve(mesh.nodes.size() * 3);
//     for (const Node* node : mesh.nodes) {
//         vertices.push_back(static_cast<float>(node->x[0]));
//         vertices.push_back(static_cast<float>(node->x[1]));
//         vertices.push_back(static_cast<float>(node->x[2]));
//     }
//
//     float* vb = (float*)rtcSetNewGeometryBuffer(
//         geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), mesh.nodes.size());
//     memcpy(vb, vertices.data(), vertices.size() * sizeof(float));
//
//     // Prepare index buffer
//     std::vector<unsigned> indices;
//     indices.reserve(mesh.faces.size() * 3);
//     for (const Face* face : mesh.faces) {
//         for (int v = 0; v < 3; ++v) {
//             indices.push_back(static_cast<unsigned>(face->v[v]->node->index));
//         }
//     }
//     unsigned* ib = (unsigned*)rtcSetNewGeometryBuffer(
//         geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), mesh.faces.size());
//     memcpy(ib, indices.data(), indices.size() * sizeof(unsigned));
//
//     rtcCommitGeometry(geom);
//     rtcAttachGeometry(scene, geom);
//     rtcReleaseGeometry(geom);
//     rtcCommitScene(scene);
//     return scene;
// }

// struct OverlapCollector {
//     const Mesh *meshB;
//     const Face *faceA;
//     const float* minA, *maxA;
//     const float* minB, *maxB;
//     BVHCallback *callback;
// };

void create_face_aabb(const Face *face, float minA[3], float maxA[3]) {
    for (int i = 0; i < 3; i++) {
        minA[i] = std::numeric_limits<float>::max();
        maxA[i] = -std::numeric_limits<float>::max();
    }
    for (int v = 0; v < 3; ++v) {
        for (int d = 0; d < 3; ++d) {
            float val = static_cast<float>(face->v[v]->node->x[d]);
            if (val < minA[d]) minA[d] = val;
            if (val > maxA[d]) maxA[d] = val;
        }
    }
}

// inline bool aabb_overlap(const float minA[3], const float maxA[3],
//                          const float minB[3], const float maxB[3]) {
//     for (int d = 0; d < 3; ++d) {
//         if (maxA[d] < minB[d] || minA[d] > maxB[d]) {
//             return false;
//         }
//     }
//     return true;
// }

// int minA;

// void mesh_mesh_overlap(const Mesh *meshA, RTCScene sceneB, const Mesh *meshB, BVHCallback callback,
//                        double thickness, bool parallel)
// {
//     int nthreads = omp_get_max_threads();
//     omp_set_num_threads(parallel ? nthreads : 1);
//     std::cout << "Using " << nthreads << " threads for " << meshA->faces.size() <<  " faces mesh-mesh overlap." << std::endl;
//
//     // Precompute AABB for meshA and meshB
//     float* minA(new float[3 * meshA->faces.size()]);
//     float* maxA(new float[3 * meshA->faces.size()]);
//     float* minB(new float[3 * meshB->faces.size()]);
//     float* maxB(new float[3 * meshB->faces.size()]);
//
// #pragma omp parallel for schedule(auto)
//     for (auto i = 0; i < meshA->faces.size(); ++i) {
//         const Face *faceA = meshA->faces[i];
//         create_face_aabb(faceA, minA + (i * 3), maxA + (i * 3));
//     }
//
// #pragma omp parallel for schedule(auto)
//     for (auto i = 0; i < meshB->faces.size(); ++i) {
//         const Face *faceB = meshB->faces[i];
//         create_face_aabb(faceB, minB + (i * 3), maxB + (i * 3));
//     }
//
//
// #pragma omp parallel for schedule(auto)
//     for (size_t i = 0; i < meshA->faces.size(); ++i) {
//         const Face *faceA = meshA->faces[i];
//         // Compute centroid of faceA
//         float cx = 0, cy = 0, cz = 0;
//         for (int v = 0; v < 3; ++v) {
//             cx += static_cast<float>(faceA->v[v]->node->x[0]);
//             cy += static_cast<float>(faceA->v[v]->node->x[1]);
//             cz += static_cast<float>(faceA->v[v]->node->x[2]);
//         }
//         cx /= 3.0f;
//         cy /= 3.0f;
//         cz /= 3.0f;
//         // Set up point query
//         RTCPointQuery query;
//         query.x = cx;
//         query.y = cy;
//         query.z = cz;
//         query.radius = static_cast<float>(thickness); // Use thickness as search radius
//         query.time = 0.0f;
//         OverlapCollector collector = {meshB, faceA, minA, maxA, minB, maxB, &callback};
//         RTCPointQueryContext context;
//         rtcInitPointQueryContext(&context);
//         auto pointQueryFunc = [](RTCPointQueryFunctionArguments *args) -> bool {
//             OverlapCollector *c = (OverlapCollector *) args->userPtr;
//             unsigned primID = args->primID;
//             const Face *faceB = c->meshB->faces[primID];
//             if (c->faceA == faceB) {
//                 return true; // Skip self-collision
//             }
//             // Check AABB overlap
//             if (aabb_overlap(c->minA + (c->faceA->index * 3), c->maxA + (c->faceA->index * 3),
//                              c->minB + (primID * 3), c->maxB + (primID * 3))) {
//                 // If AABBs overlap, call the callback
//                 (*(c->callback))(c->faceA, faceB);
//             }
//             return true; // continue search
//         };
//         rtcPointQuery(sceneB, &query, &context, pointQueryFunc, &collector);
//     }
//     omp_set_num_threads(nthreads);
// }

// Embree-accelerated version of for_overlapping_faces
// void for_overlapping_faces_embree2(
//     const std::vector<Mesh*>& meshes,
//     const std::vector<Mesh*>& obs_meshes,
//     double thickness,
//     BVHCallback callback,
//     bool parallel,
//     bool only_obs)
// {
//     RTCDevice device = rtcNewDevice(nullptr);
//     std::vector<RTCScene> meshScenes, obsScenes;
//     for (auto mesh : meshes) {
//         meshScenes.push_back(create_embree_scene_from_mesh(*mesh, device));
//     }
//     for (auto mesh : obs_meshes) {
//         obsScenes.push_back(create_embree_scene_from_mesh(*mesh, device));
//     }
//
//     for (int i = 0; i < (int)meshScenes.size(); ++i) {
//         if (!only_obs) {
//             // Self-collision: mesh i with itself and with previous meshes
//             mesh_mesh_overlap(meshes[i], meshScenes[i], meshes[i], callback,
//                 thickness, parallel);
//             for (int j = 0; j < i; ++j) {
//                 mesh_mesh_overlap(meshes[i], meshScenes[j], meshes[j], callback,
//                 thickness, parallel);
//             }
//         }
//         // Cloth-obstacle collision
//         for (int o = 0; o < (int)obsScenes.size(); ++o) {
//             mesh_mesh_overlap(meshes[i], obsScenes[o], obs_meshes[o], callback,
//                 thickness, parallel);
//         }
//     }
//
//     for (auto scene : meshScenes) rtcReleaseScene(scene);
//     for (auto scene : obsScenes) rtcReleaseScene(scene);
//     rtcReleaseDevice(device);
// }

float embree_global_thickness = 0.0f;

void boundsFunc(const RTCBoundsFunctionArguments *args) {
    float thickness = embree_global_thickness;
    Face* face = *(static_cast<Face**>(args->geometryUserPtr) + args -> primID);
    float minA[3], maxA[3];

    create_face_aabb(face, minA, maxA);

    args->bounds_o -> lower_x = minA[0] - thickness;
    args->bounds_o -> lower_y = minA[1] - thickness;
    args->bounds_o -> lower_z = minA[2] - thickness;
    args->bounds_o -> upper_x = maxA[0] + thickness;
    args->bounds_o -> upper_y = maxA[1] + thickness;
    args->bounds_o -> upper_z = maxA[2] + thickness;
}


RTCScene build_aabb_mesh_scene(Mesh* mesh, RTCDevice device, double thickness) {
    RTCScene scene = rtcNewScene(device);
    void *face0_ptr = mesh->faces.data();
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_USER);
    rtcSetGeometryUserPrimitiveCount(geom, mesh->faces.size());
    rtcSetGeometryUserData(geom, face0_ptr);

    embree_global_thickness = thickness / 2;

    rtcSetGeometryBoundsFunction(geom, boundsFunc, nullptr);
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);
    return scene;
}


struct CollideCollector {
    RTCScene scene0, scene1;
    BVHCallback* callback;
};

void collideCallBack(void* userPtr, RTCCollision* collisions, unsigned int num_collisions) {
    CollideCollector* c = (CollideCollector*)userPtr;

    for (size_t i = 0; i < num_collisions; ++i) {
        RTCGeometry geom0 = rtcGetGeometry(c->scene0, collisions[i].geomID0);
        RTCGeometry geom1 = rtcGetGeometry(c->scene1, collisions[i].geomID1);

        Face* face0 = *(static_cast<Face**>(rtcGetGeometryUserData(geom0)) + collisions[i].primID0);
        Face* face1 = *(static_cast<Face**>(rtcGetGeometryUserData(geom1)) + collisions[i].primID1);

        if (face0 == face1) {
            continue; // Skip self-collision
        }
        (*(c->callback))(face0, face1);
    }
}


void overlap_meshes(RTCScene sceneA, const Mesh* meshA, RTCScene sceneB, const Mesh* meshB,
    BVHCallback callback) {
    CollideCollector collector = { sceneA, sceneB, &callback };
    rtcCollide(sceneA, sceneB, collideCallBack, &collector);
}


void for_overlapping_faces_embree(const std::vector<Mesh*>& meshes,
    const std::vector<Mesh*>& obs_meshes,
    double thickness,
    BVHCallback callback,
    bool parallel,
    bool only_obs)
{
    int embree_num_threads = 4;
    if (const char* env_p = std::getenv("EMBREE_NUM_THREADS")) {
        try {
            int num_threads = std::stoi(env_p);
            if (num_threads > 0) embree_num_threads = num_threads;
        } catch (...) {}
    }
    std::string device_config = "threads=" + std::to_string(embree_num_threads) + ",verbose=0 user_threads=" + std::to_string(embree_num_threads);
    RTCDevice device = rtcNewDevice(device_config.c_str());
    std::vector<RTCScene> meshScenes, obsScenes;
    for (auto mesh : meshes) {
        meshScenes.push_back(build_aabb_mesh_scene(mesh, device, thickness));
    }
    for (auto mesh : obs_meshes) {
        if (mesh->faces.empty()) {
            continue; // Skip empty meshes
        }
        obsScenes.push_back(build_aabb_mesh_scene(mesh, device, thickness));
    }

    for (int i = 0; i < (int)meshScenes.size(); ++i) {
        if (!only_obs) {
            // Self-collision: mesh i with itself and with previous meshes
            for (int j = 0; j <= i; ++j) {
                overlap_meshes(meshScenes[i], meshes[i], meshScenes[j], meshes[j], callback);
            }
        }
        // Cloth-obstacle collision
        for (int o = 0; o < (int)obsScenes.size(); ++o) {
            overlap_meshes(meshScenes[i], meshes[i], obsScenes[o], obs_meshes[o], callback);
        }
    }

    for (auto scene : meshScenes) {
        rtcReleaseScene(scene);
    }
    for (auto scene : obsScenes) {
        rtcReleaseScene(scene);
    }
    rtcReleaseDevice(device);
}

// Create EmbreeAccel from Mesh
// EmbreeAccel create_embree_accel_from_mesh(const Mesh& mesh, RTCDevice device) {
//     EmbreeAccel acc;
//     acc.scene = rtcNewScene(device);
//     acc.geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
//     acc.num_vertices = mesh.nodes.size();
//     // Prepare vertex buffer
//     acc.vertex_buffer = (float*)rtcSetNewGeometryBuffer(
//         acc.geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), mesh.nodes.size());
//     for (size_t i = 0; i < mesh.nodes.size(); ++i) {
//         acc.vertex_buffer[3*i+0] = static_cast<float>(mesh.nodes[i]->x[0]);
//         acc.vertex_buffer[3*i+1] = static_cast<float>(mesh.nodes[i]->x[1]);
//         acc.vertex_buffer[3*i+2] = static_cast<float>(mesh.nodes[i]->x[2]);
//     }
//     // Prepare index buffer
//     std::vector<unsigned> indices;
//     indices.reserve(mesh.faces.size() * 3);
//     for (const Face* face : mesh.faces) {
//         for (int v = 0; v < 3; ++v)
//             indices.push_back(static_cast<unsigned>(face->v[v]->node->index));
//     }
//     unsigned* ib = (unsigned*)rtcSetNewGeometryBuffer(
//         acc.geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), mesh.faces.size());
//     memcpy(ib, indices.data(), indices.size() * sizeof(unsigned));
//     rtcCommitGeometry(acc.geom);
//     rtcAttachGeometry(acc.scene, acc.geom);
//     rtcReleaseGeometry(acc.geom);
//     rtcCommitScene(acc.scene);
//     return acc;
// }
//
// // Update EmbreeAccel after mesh vertex positions change
// void update_embree_accel(const Mesh& mesh, EmbreeAccel& acc) {
//     for (size_t i = 0; i < mesh.nodes.size(); ++i) {
//         acc.vertex_buffer[3*i+0] = static_cast<float>(mesh.nodes[i]->x[0]);
//         acc.vertex_buffer[3*i+1] = static_cast<float>(mesh.nodes[i]->x[1]);
//         acc.vertex_buffer[3*i+2] = static_cast<float>(mesh.nodes[i]->x[2]);
//     }
//     rtcCommitGeometry(acc.geom);
//     rtcCommitScene(acc.scene);
// }
//
// // Destroy EmbreeAccel
// void destroy_embree_accel(EmbreeAccel& acc) {
//     if (acc.scene) rtcReleaseScene(acc.scene);
//     acc.scene = nullptr;
//     acc.geom = nullptr;
//     acc.vertex_buffer = nullptr;
//     acc.num_vertices = 0;
// }
