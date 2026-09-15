#include "gaffa/ascend_runtime.h"
#include "gaffa/vector_add.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(VectorAdd, EmptyInputs) {
  EXPECT_TRUE(gaffa::vector_add({}, {}).empty());
}

TEST(VectorAdd, RejectsMismatchedInputSizes) {
  EXPECT_THROW((void)gaffa::vector_add({1.0F}, {1.0F, 2.0F}),
               std::invalid_argument);
}

TEST(VectorAdd, HandlesSingleElementOnAscendDevice) {
  if (gaffa::ascend_device_count() == 0) {
    GTEST_SKIP() << "Ascend device is not visible";
  }

  EXPECT_EQ(gaffa::vector_add({1.0F}, {2.0F}),
            (std::vector<float>{3.0F}));
}

class VectorAddAscendSizes : public testing::TestWithParam<int> {};

TEST_P(VectorAddAscendSizes, AddsOnAscendDevice) {
  if (gaffa::ascend_device_count() == 0) {
    GTEST_SKIP() << "Ascend device is not visible";
  }

  const int size = GetParam();
  std::vector<float> lhs(size);
  std::vector<float> rhs(size);
  std::vector<float> expected(size);
  for (int index = 0; index < size; ++index) {
    lhs[index] = static_cast<float>(index);
    rhs[index] = static_cast<float>(size - index);
    expected[index] = static_cast<float>(size);
  }

  EXPECT_EQ(gaffa::vector_add(lhs, rhs), expected);
}

INSTANTIATE_TEST_SUITE_P(
    BoundarySizes,
    VectorAddAscendSizes,
    testing::Values(255, 256, 257, 1024));
