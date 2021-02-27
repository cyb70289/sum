#include <cmath>
#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

// naive summation suffers from round-off error
static double sum_naive(const double* value, size_t len) {
  double sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += value[i];
  }
  return sum;
}

// recursive pairwise summation with higher precision
static double sum_pairwise_recur(const double* value, size_t len) {
  // minimal block size to do divide and conqure
  // smaller value -> better precision, larger value -> better performance
  constexpr int min_block_size = 16;  // same as numpy
  if (len <= min_block_size) {
    return sum_naive(value, len);
  }
  return sum_pairwise_recur(value, len/2) +
         sum_pairwise_recur(value + len/2, len - len/2);
}

// non-recursive pariwise summation
static double sum_pairwise_itera(const double* value, size_t len) {
  constexpr int min_block_size = 16;
  static_assert(min_block_size > 1, "min_block_size must be greater than 1");

  // levels (tree depth) = ceil(log2(len)), a bit larger than necessary
  const int levels = sizeof(len) * 8 - __builtin_clzll(len);

  // temporary summation per level
  std::vector<double> sum(levels);
  // whether two summations are ready and should be reduced to upper level
  // one bit for each level, bit0 -> level0, ...
  uint64_t mask = 0;
  // level of root node holding the final summation
  int root_level = 0;

  for (size_t i = 0; i < len / min_block_size; ++i) {
    // temporary summation for one block
    double tmpsum = 0;
    for (int j = 0; j < min_block_size; ++j) {
      tmpsum += value[j];
    }
    value += min_block_size;

    // reduce to upper level if two summations are ready
    int cur_level = 0;
    uint64_t cur_level_mask = 1 << cur_level;
    sum[cur_level] += tmpsum;
    mask ^= cur_level_mask;
    while ((mask & cur_level_mask) == 0) {
      tmpsum = sum[cur_level];
      sum[cur_level] = 0;
      ++cur_level;
      cur_level_mask <<= 1;
      sum[cur_level] += tmpsum;
      mask ^= cur_level_mask;
    }
    root_level = std::max(root_level, cur_level);
  }

  // remaining data
  double tmpsum = 0;
  for (size_t i = 0; i < len % min_block_size; ++i) {
    tmpsum += value[i];
  }

  // reduce temporay summations
  sum[0] += tmpsum;
  for (int i = 1; i <= root_level; ++i) {
    sum[i] += sum[i - 1];
  }

  return sum[root_level];
}

static const std::vector<double> GenerateTestData(size_t len) {
  std::vector<double> v(len);
  std::iota(v.begin(), v.end(), 0);   // 0, 1, 2, ...
  return v;
}

static void BM_naive(benchmark::State& state) {
  const std::vector<double> value = GenerateTestData(state.range(0));

  for (auto _ : state) {
    benchmark::DoNotOptimize(sum_naive(value.data(), value.size()));
  }
}

static void BM_pairwise_recur(benchmark::State& state) {
  const std::vector<double> value = GenerateTestData(state.range(0));

  for (auto _ : state) {
    benchmark::DoNotOptimize(sum_pairwise_recur(value.data(), value.size()));
  }
}

static void BM_pairwise_itera(benchmark::State& state) {
  const std::vector<double> value = GenerateTestData(state.range(0));

  for (auto _ : state) {
    benchmark::DoNotOptimize(sum_pairwise_itera(value.data(), value.size()));
  }
}

BENCHMARK(BM_naive)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_pairwise_recur)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_pairwise_itera)->Arg(4096)->Arg(65536)->Arg(1 << 20);

TEST(Sum, Basic) {
  const std::vector<double> value = GenerateTestData(1000);
  const double sum = 999 * 500;
  EXPECT_EQ(sum_naive(value.data(), value.size()), sum);
  EXPECT_EQ(sum_pairwise_recur(value.data(), value.size()), sum);
  EXPECT_EQ(sum_pairwise_itera(value.data(), value.size()), sum);
}

// naive sum fails this test
TEST(Sum, Roundoff) {
  std::vector<double> value(321000);
  for (int i = 0; i < 321000; ++i) {
    value[i] = (i - 160499.5) * (i - 160499.5);
  }

  EXPECT_EQ(sum_pairwise_recur(value.data(), value.size()), 2756346749973250);
  EXPECT_EQ(sum_pairwise_itera(value.data(), value.size()), 2756346749973250);
}

// mix test and benchmark, may corrupt argc/argv
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  if (ret == 0) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
  }
  return ret;
}
