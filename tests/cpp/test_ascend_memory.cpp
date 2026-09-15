#include "gaffa/ascend_memory.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

static_assert(std::is_move_constructible_v<gaffa::AscendDeviceMemory>);
static_assert(!std::is_copy_constructible_v<gaffa::AscendDeviceMemory>);

TEST(AscendMemory, SpanReportsShapeBytesAndDevice) {
  int values[4] = {1, 2, 3, 4};
  const gaffa::AscendSpan<int> span{
      .data = values,
      .count = 4,
      .device_id = 3,
  };

  EXPECT_EQ(span.size(), 4U);
  EXPECT_EQ(span.bytes(), 4U * sizeof(int));
  EXPECT_FALSE(span.empty());
  EXPECT_EQ(span.device_id, 3);

  const gaffa::AscendSpan<int> empty{};
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.bytes(), 0U);
}

TEST(AscendMemory, CheckedBufferBytesRejectsOverflow) {
  EXPECT_EQ(gaffa::detail::checked_ascend_buffer_bytes<std::uint16_t>(7), 14U);

  const auto too_many =
      std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) + 1;
  EXPECT_THROW(
      (void)gaffa::detail::checked_ascend_buffer_bytes<std::uint32_t>(too_many),
      std::overflow_error);
}
