#include "gaffa/vector_add.hpp"

#include "gaffa/vector_add_ascend.hpp"

#include <stdexcept>
#include <vector>

namespace gaffa {

std::vector<float> vector_add(const std::vector<float>& lhs,
                              const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("vector_add requires inputs with the same length");
  }
  if (lhs.empty()) return {};
  return vector_add_ascend(lhs, rhs, 0);
}

}  // namespace gaffa
