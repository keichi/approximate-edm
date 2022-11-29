#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#ifdef __NEC__
#include <asl.h>
#endif

#include "timer.hpp"

const int L = 10000;
const int E = 20;
const int tau = 1;
const int Tp = 1;
const int top_k = E + 1;

const int iterations = 10;

template <class T> class Counter
{
private:
    T i_;

public:
    using difference_type = T;
    using value_type = T;
    using pointer = T;
    using reference = T &;
    using iterator_category = std::input_iterator_tag;

    explicit Counter(T i) : i_(i) {}
    T operator*() const noexcept { return i_; }
    Counter &operator++() noexcept
    {
        i_++;
        return *this;
    }
    bool operator==(const Counter &rhs) const { return i_ == rhs.i_; }
    bool operator!=(const Counter &rhs) const { return i_ != rhs.i_; }
};

void calc_distances(std::vector<float> &distances,
                    const std::vector<float> &library,
                    const std::vector<float> &target)
{
    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

#pragma omp parallel for
    for (int i = 0; i < n_target; i++) {
        for (int j = 0; j < n_library; j++) {
            distances[i * n_library + j] = 0.0f;
        }

        for (int k = 0; k < E; k++) {
            float tmp = target[i + k * tau];

            for (int j = 0; j < n_target; j++) {
                float diff = tmp - library[j + k * tau];
                distances[i * n_library + j] += diff * diff;
            }
        }
    }
}

void full_sort_stl(std::vector<float> &distances, std::vector<int> &indices,
                   const std::vector<float> &library,
                   const std::vector<float> &target)
{
    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

#pragma omp parallel for
    for (int i = 0; i < n_target; i++) {
        std::iota(indices.begin() + i * n_library,
                  indices.begin() + (i + 1) * n_library, 0);

        std::sort(indices.begin() + i * n_library,
                  indices.begin() + (i + 1) * n_library,
                  [&](int a, int b) -> int {
                      return distances[i * n_library + a] <
                             distances[i * n_library + b];
                  });
    }
}

void partial_sort_stl(std::vector<float> &distances, std::vector<int> &indices,
                      const std::vector<float> &library,
                      const std::vector<float> &target)
{
    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

#pragma omp parallel for
    for (int i = 0; i < n_target; i++) {
        std::partial_sort_copy(Counter<int>(0), Counter<int>(n_library),
                               indices.begin() + i * n_library,
                               indices.begin() + i * n_library + top_k,
                               [&](int a, int b) -> int {
                                   return distances[i * n_library + a] <
                                          distances[i * n_library + b];
                               });
    }
}

#ifdef __NEC__
void full_sort_asl(std::vector<float> &distances, std::vector<int> &indices,
                   const std::vector<float> &library,
                   const std::vector<float> &target)
{
    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

    asl_sort_t sort;

    asl_library_initialize();

    asl_sort_create_s(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    asl_sort_preallocate(sort, n_library);

#pragma omp parallel for
    for (int i = 0; i < n_target; i++) {
        asl_sort_execute_s(sort, n_library, &distances[i * n_library], ASL_NULL,
                           &distances[i * n_library], &indices[i * n_library]);
    }

    asl_sort_destroy(sort);
    asl_library_finalize();
}
#endif

int main(int argc, char *argv[])
{
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    std::vector<float> library(L);
    std::vector<float> target(L);

    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

    std::vector<float> distances(n_library * n_target);
    std::vector<int> indices(n_library * n_target);

    for (int i = 0; i < L; i++) {
        library[i] = dist(engine);
        target[i] = dist(engine);
    }

    for (int iter = 0; iter < iterations; iter++) {
        Timer timer_dist, timer_sort;

        timer_dist.start();

#pragma _NEC noinline
        calc_distances(distances, library, target);

        timer_dist.stop();
        timer_sort.start();

        // full_sort_stl(distances, indices, library, target);

        partial_sort_stl(distances, indices, library, target);

#pragma _NEC noinline
        full_sort_asl(distances, indices, library, target);

        timer_sort.stop();

        std::cout << "parwise distances " << timer_dist.elapsed() << " [ms]"
                  << std::endl;
        std::cout << "top-k " << timer_sort.elapsed() << " [ms]"
                  << std::endl;
    }

    return 0;
}
