#include <cstddef>
#include <iostream>
#include <vector>

#include <xtensor.hpp>

#include <faiss/index_factory.h>

#include "lorenz.hpp"
#include "timer.hpp"

const int tau = 1;
const int Tp = 1;

const int N_WARMUPS = 3;
const int N_TRIALS = 10;

struct Timers {
    Timer total;
    Timer train;
    Timer index;
    Timer search;
};

void simplex(faiss::Index *index, const xt::xtensor<float, 1> &train,
             const xt::xtensor<float, 1> &test, int E, Timers &timers,
             bool measure_timings)
{
    xt::xtensor<float, 2> train_embed = xt::empty<float>(
        {train.shape(0) - (E - 1) * tau, static_cast<size_t>(E)});
    xt::xtensor<float, 2> test_embed = xt::empty<float>(
        {test.shape(0) - (E - 1) * tau, static_cast<size_t>(E)});

    // Embedded train and test time series
    for (int i = 0; i < E; i++) {
        xt::col(train_embed, i) = xt::view(
            train, xt::range(i * tau, train.size() - (E - i - 1) * tau, tau));
        xt::col(test_embed, i) = xt::view(
            test, xt::range(i * tau, test.size() - (E - i - 1) * tau, tau));
    }

    xt::xtensor<float, 2> dist =
        xt::empty<float>({test_embed.shape(0), static_cast<size_t>(E + 1)});
    xt::xtensor<faiss::Index::idx_t, 2> ind = xt::empty<faiss::Index::idx_t>(
        {test_embed.shape(0), static_cast<size_t>(E + 1)});

    if (measure_timings) {
        timers.total.start();
        timers.train.start();
    }

    index->train(train_embed.shape(0), train_embed.data());

    if (measure_timings) {
        timers.train.stop();
        timers.index.start();
    }

    index->add(train_embed.shape(0) - Tp, train_embed.data());

    if (measure_timings) {
        timers.index.stop();
        timers.search.start();
    }

    index->search(test_embed.shape(0), test_embed.data(), E + 1, dist.data(),
                  ind.data());

    if (measure_timings) {
        timers.search.stop();
        timers.total.stop();
    }
}

int main(int argc, char *argv[])
{
    std::cout << "Index: " << argv[1] << std::endl;
    std::cout << "E\tN\ttotal\ttrain\tindex\tsearch" << std::endl;

    for (int E : std::vector<int>({1, 5, 10, 20})) {
        int start = std::stoi(argv[2]);
        int end = std::stoi(argv[3]);

        for (int N = 1 << start; N <= 1 << end; N <<= 1) {
            Timers timers;

            std::cout << E << "\t" << N << "\t";

            xt::xtensor<float, 1> ts = xt::empty<float>({N});

            generate_lorenz(ts, 40.0f / ts.size());

            auto train = xt::view(ts, xt::range(0, ts.size() / 2));
            auto test = xt::view(ts, xt::range(ts.size() / 2, ts.size()));

            for (int i = 0; i < N_WARMUPS + N_TRIALS; i++) {
                faiss::Index *index = faiss::index_factory(E, argv[1]);
                simplex(index, train, test, E, timers, i >= N_WARMUPS);
            }

            std::cout << std::fixed << std::setprecision(2)
                      << timers.total.elapsed() / N_TRIALS << "\t"
                      << timers.train.elapsed() / N_TRIALS << "\t"
                      << timers.index.elapsed() / N_TRIALS << "\t"
                      << timers.search.elapsed() / N_TRIALS << std::endl;
        }
    }

    return 0;
}
