#include <vector>
#include <Eigen/Core>
#include <mesh.hpp>

// bool load_frames_binary(const std::string& filename, std::vector<Eigen::MatrixX3f>& frames);
bool load_frames(const std::string &filename, std::vector<Eigen::MatrixX3f>& frames);
void save_mesh_state(const Mesh& mesh, const std::string& filename);
bool load_mesh_state(Mesh &mesh, const std::string& filename, const std::string& baseline_filename="");