#ifndef ARCSIM_QP_HPP
#define ARCSIM_QP_HPP

#include "optimization.hpp"
#include "collision.hpp"

void projected_gauss_seidel (ImpactZone *zone, std::unordered_map<const Node*, Vec3> &xold);

#endif //ARCSIM_QP_HPP