#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace {

// naive summation suffers from round-off error
double sum_naive(const double* value, size_t len) {
  double sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += value[i];
  }
  return sum;
}

// recursive pairwise summation with higher precision
double sum_pairwise_recur(const double* value, size_t len) {
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
double sum_pairwise_itera(const double* value, size_t len) {
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

struct KahanSum {
  double sum = 0;
  double c = 0;

  KahanSum& operator+=(double v) {
    double y = v - (std::isnan(c) ? 0 : c);
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
    return *this;
  }

  KahanSum& operator+=(const KahanSum& other) {
    (*this) += other.sum;
    (*this) += (std::isnan(other.c) ? 0 : other.c);
    return *this;
  }

  double operator()() {
    return sum;
  }
};

double sum_kahan_naive(const double* value, size_t len) {
  KahanSum sum;
  for (size_t i = 0; i < len; ++i) {
    sum += value[i];
  }
  return sum();
}

double sum_kahan_interleaved(const double* value, size_t len) {
  constexpr int kInterleave = 8;

  std::array<KahanSum, kInterleave> sums;

  size_t i = 0;
  for (; i < len - kInterleave; i += kInterleave) {
    for (size_t j = 0; j < kInterleave; ++j) {
      sums[j] += value[i + j];
    }
  }
  for (; i < len; ++i) {
    sums[0] += value[i];
  }
  for (size_t j = 1; j < kInterleave; ++j) {
    sums[0] += sums[j];
  }
  return sums[0]();
}

double sum_kahan_blocked(const double* value, size_t len) {
  // Just like sum_pairwise_itera and sum_pairwise_recur, this trades off
  // a bit of precision (by allowing for errors to accumulate inside a single
  // block) in exchange of higher performance.
  constexpr int kBlockSize = 16;
  constexpr int kInterleave = 8;

  std::array<KahanSum, kInterleave> sums;

  size_t i = 0;
  for (; i < len - (kBlockSize * kInterleave); i += (kBlockSize * kInterleave)) {
    std::array<double, kInterleave> block_sums{};
    for (size_t j = 0; j < kBlockSize; ++j) {
      for (size_t k = 0; k < kInterleave; ++k) {
        block_sums[k] += value[i + j * kInterleave + k];
      }
    }
    for (size_t j = 0; j < kInterleave; ++j) {
      sums[j] += block_sums[j];
    }
  }
  for (; i < len - kBlockSize; i += kBlockSize) {
    double block_sum = 0;
    for (size_t j = 0; j < kBlockSize; ++j) {
      block_sum += value[i + j];
    }
    sums[0] += block_sum;
  }
  for (; i < len; ++i) {
    sums[0] += value[i];
  }
  for (size_t j = 1; j < kInterleave; ++j) {
    sums[0] += sums[j];
  }
  return sums[0]();
}

struct NeumaierSum {
  double sum = 0;
  double c = 0;

  NeumaierSum& operator+=(double v) {
    double t = sum + v;
    c += (std::abs(sum) >= std::abs(v)) ? (sum - t) + v : (v - t) + sum;
    sum = t;
    return *this;
  }

  NeumaierSum& operator+=(const NeumaierSum& other) {
    (*this) += other.sum;
    (*this) += (std::isnan(other.c) ? 0 : other.c);
    return *this;
  }

  double operator()() {
    return sum + (std::isnan(c) ? 0 : c);
  }
};

double sum_neumaier_naive(const double* value, size_t len) {
  NeumaierSum sum;
  for (size_t i = 0; i < len; ++i) {
    sum += value[i];
  }
  return sum();
}

double sum_neumaier_interleaved(const double* value, size_t len) {
  constexpr int kInterleave = 4;

  std::array<NeumaierSum, kInterleave> sums;

  size_t i = 0;
  for (; i < len - kInterleave; i += kInterleave) {
    for (size_t j = 0; j < kInterleave; ++j) {
      sums[j] += value[i + j];
    }
  }
  for (; i < len; ++i) {
    sums[0] += value[i];
  }
  for (size_t j = 1; j < kInterleave; ++j) {
    sums[0] += sums[j];
  }
  return sums[0]();
}

double sum_neumaier_blocked(const double* value, size_t len) {
  // Similar to sum_kahan_blocked
  constexpr int kBlockSize = 16;
  constexpr int kInterleave = 8;

  std::array<NeumaierSum, kInterleave> sums;

  size_t i = 0;
  for (; i < len - (kBlockSize * kInterleave); i += (kBlockSize * kInterleave)) {
    std::array<double, kInterleave> block_sums{};
    for (size_t j = 0; j < kBlockSize; ++j) {
      for (size_t k = 0; k < kInterleave; ++k) {
        block_sums[k] += value[i + j * kInterleave + k];
      }
    }
    for (size_t j = 0; j < kInterleave; ++j) {
      sums[j] += block_sums[j];
    }
  }
  for (; i < len - kBlockSize; i += kBlockSize) {
    double block_sum = 0;
    for (size_t j = 0; j < kBlockSize; ++j) {
      block_sum += value[i + j];
    }
    sums[0] += block_sum;
  }
  for (; i < len; ++i) {
    sums[0] += value[i];
  }
  for (size_t j = 1; j < kInterleave; ++j) {
    sums[0] += sums[j];
  }
  return sums[0]();
}

const std::vector<double> GenerateTestData(size_t len) {
  std::vector<double> v(len);
  std::iota(v.begin(), v.end(), 0);   // 0, 1, 2, ...
  return v;
}

template <typename SumFunc>
void BenchmarkSummer(benchmark::State& state, SumFunc&& sum_func) {
  const std::vector<double> value = GenerateTestData(state.range(0));

  for (auto _ : state) {
    benchmark::DoNotOptimize(sum_func(value.data(), value.size()));
  }
  state.SetItemsProcessed(state.iterations() * value.size());
}

void BM_naive(benchmark::State& state) {
  BenchmarkSummer(state, sum_naive);
}

void BM_pairwise_recur(benchmark::State& state) {
  BenchmarkSummer(state, sum_pairwise_recur);
}

void BM_pairwise_itera(benchmark::State& state) {
  BenchmarkSummer(state, sum_pairwise_itera);
}

void BM_kahan_naive(benchmark::State& state) {
  BenchmarkSummer(state, sum_kahan_naive);
}

void BM_kahan_interleaved(benchmark::State& state) {
  BenchmarkSummer(state, sum_kahan_interleaved);
}

void BM_kahan_blocked(benchmark::State& state) {
  BenchmarkSummer(state, sum_kahan_blocked);
}

void BM_neumaier_naive(benchmark::State& state) {
  BenchmarkSummer(state, sum_neumaier_naive);
}

void BM_neumaier_interleaved(benchmark::State& state) {
  BenchmarkSummer(state, sum_neumaier_interleaved);
}

void BM_neumaier_blocked(benchmark::State& state) {
  BenchmarkSummer(state, sum_neumaier_blocked);
}

BENCHMARK(BM_naive)->Arg(128)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_pairwise_recur)->Arg(128)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_pairwise_itera)->Arg(128)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_kahan_naive)->Arg(128)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_kahan_interleaved)->Arg(128)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_kahan_blocked)->Arg(128)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_neumaier_naive)->Arg(128)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_neumaier_interleaved)->Arg(128)->Arg(4096)->Arg(65536)->Arg(1 << 20);
BENCHMARK(BM_neumaier_blocked)->Arg(128)->Arg(4096)->Arg(65536)->Arg(1 << 20);

TEST(Sum, Basic) {
  const std::vector<double> value = GenerateTestData(1000);
  const double sum = 999 * 500;
  EXPECT_EQ(sum_naive(value.data(), value.size()), sum);
  EXPECT_EQ(sum_pairwise_recur(value.data(), value.size()), sum);
  EXPECT_EQ(sum_pairwise_itera(value.data(), value.size()), sum);
  EXPECT_EQ(sum_kahan_naive(value.data(), value.size()), sum);
  EXPECT_EQ(sum_kahan_interleaved(value.data(), value.size()), sum);
  EXPECT_EQ(sum_kahan_blocked(value.data(), value.size()), sum);
  EXPECT_EQ(sum_neumaier_naive(value.data(), value.size()), sum);
  EXPECT_EQ(sum_neumaier_interleaved(value.data(), value.size()), sum);
  EXPECT_EQ(sum_neumaier_blocked(value.data(), value.size()), sum);
}

// naive sum fails this test
TEST(Sum, Roundoff) {
  std::vector<double> value(321000);
  for (int i = 0; i < 321000; ++i) {
    value[i] = (i - 160499.5) * (i - 160499.5);
  }

  EXPECT_EQ(sum_pairwise_recur(value.data(), value.size()), 2756346749973250);
  EXPECT_EQ(sum_pairwise_itera(value.data(), value.size()), 2756346749973250);
  EXPECT_EQ(sum_kahan_naive(value.data(), value.size()), 2756346749973250);
  EXPECT_EQ(sum_kahan_interleaved(value.data(), value.size()), 2756346749973250);
  EXPECT_EQ(sum_kahan_blocked(value.data(), value.size()), 2756346749973250);
  EXPECT_EQ(sum_neumaier_naive(value.data(), value.size()), 2756346749973250);
  EXPECT_EQ(sum_neumaier_interleaved(value.data(), value.size()), 2756346749973250);
  EXPECT_EQ(sum_neumaier_blocked(value.data(), value.size()), 2756346749973250);
}

// Only pure Neumaier sums pass this test
TEST(Sum, Roundoff2) {
  std::vector<double> value;
  for (int i = 0; i < 100; ++i) {
    value.push_back(1);
    value.push_back(1e100);
    value.push_back(1);
    value.push_back(-1e100);
    value.push_back(1);
  }

  EXPECT_EQ(sum_neumaier_naive(value.data(), value.size()), 300);
  EXPECT_EQ(sum_neumaier_interleaved(value.data(), value.size()), 300);
}

TEST(Sum, NonFinite) {
  std::vector<double> value;
  for (int i = 0; i < 100; ++i) {
    value.push_back(1);
    value.push_back(HUGE_VAL);
    value.push_back(1);
  }

  EXPECT_EQ(sum_naive(value.data(), value.size()), HUGE_VAL);
  EXPECT_EQ(sum_pairwise_itera(value.data(), value.size()), HUGE_VAL);
  EXPECT_EQ(sum_pairwise_recur(value.data(), value.size()), HUGE_VAL);
  EXPECT_EQ(sum_kahan_naive(value.data(), value.size()), HUGE_VAL);
  EXPECT_EQ(sum_kahan_interleaved(value.data(), value.size()), HUGE_VAL);
  EXPECT_EQ(sum_kahan_blocked(value.data(), value.size()), HUGE_VAL);
  EXPECT_EQ(sum_neumaier_naive(value.data(), value.size()), HUGE_VAL);
  EXPECT_EQ(sum_neumaier_interleaved(value.data(), value.size()), HUGE_VAL);
  EXPECT_EQ(sum_neumaier_blocked(value.data(), value.size()), HUGE_VAL);
}

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
