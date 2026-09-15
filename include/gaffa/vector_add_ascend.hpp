#pragma once

#include <vector>

namespace gaffa {

std::vector<float> vector_add_ascend(const std::vector<float>& lhs,
                                     const std::vector<float>& rhs,
                                     int device_id = 0);

}  // namespace gaffa
