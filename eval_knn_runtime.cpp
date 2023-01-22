#include <iostream>
#include <vector>

#include <cxxopts.hpp>
#include <faiss/IndexHNSW.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_factory.h>
#include <nanoflann.hpp>
#include <xtensor.hpp>

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

template <class T> class XtensorDataset
{
    const xt::xtensor<T, 2> &dataset;

public:
    XtensorDataset(const xt::xtensor<T, 2> &dataset) : dataset(dataset) {}

    inline size_t kdtree_get_point_count() const { return dataset.shape(0); }

    inline T kdtree_get_pt(const size_t idx, int dim) const
    {
        return dataset(idx, dim);
    }

    template <class BBOX> bool kdtree_get_bbox(BBOX &bb) const { return false; }
};

void simplex(faiss::Index *_index, const xt::xtensor<float, 1> &train,
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
    xt::xtensor<unsigned int, 2> ind = xt::empty<faiss::Index::idx_t>(
        {test_embed.shape(0), static_cast<size_t>(E + 1)});

    if (measure_timings) {
        timers.total.start();
        timers.train.start();
    }

    if (measure_timings) {
        timers.train.stop();
        timers.index.start();
    }

    XtensorDataset<float> dataset(train_embed);
    nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, XtensorDataset<float>>,
        XtensorDataset<float>>
        index(E, dataset);
    index.buildIndex();

    if (measure_timings) {
        timers.index.stop();
        timers.search.start();
    }

#pragma omp parallel for
    for (int i = 0; i < test_embed.shape(0); i++) {
        index.knnSearch(
            xt::row(test_embed, i).data() +
                xt::row(test_embed, i).data_offset(),
            E + 1, xt::row(ind, i).data() + xt::row(ind, i).data_offset(),
            xt::row(dist, i).data() + xt::row(dist, i).data_offset());
    }

    if (measure_timings) {
        timers.search.stop();
        timers.total.stop();
    }
}

int main(int argc, char *argv[])
{
    cxxopts::Options options(
        "eval-knn-runtime",
        "Compare runtime of Simplex projection across AkNN algorithms");

    // clang-format off
    options.add_options()
        ("h,help", "Print usage")
        ("g,gpu", "Use GPU if index is supported")
        ("e,embedding-dims", "Embedding dimensions",
         cxxopts::value<std::vector<int>>()->default_value("1,5,10,20"))
        ("log2-min-n", "log2(minimum time series length)",
         cxxopts::value<int>()->default_value("10"))
        ("log2-max-n", "log2(maximum time series length)",
         cxxopts::value<int>()->default_value("20"))
        ("efConstruction", "efConstruction parameter for HNSW Index",
         cxxopts::value<int>()->default_value("40"))
        ("efSearch", "efSearch parameter for HNSW Index",
         cxxopts::value<int>()->default_value("16"))
        ("index", "Index factory string", cxxopts::value<std::string>());
    // clang-format on

    options.parse_positional("index");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    faiss::gpu::StandardGpuResources res;

    std::cout << std::boolalpha
              << "Index: " << result["index"].as<std::string>()
              << ", GPU: " << result["gpu"].as<bool>() << std::endl;
    std::cout << "E\tN\ttotal\ttrain\tindex\tsearch" << std::endl;

    for (int E : result["embedding-dims"].as<std::vector<int>>()) {
        int start = result["log2-min-n"].as<int>();
        int end = result["log2-max-n"].as<int>();

        for (int N = 1 << start; N <= 1 << end; N <<= 1) {
            Timers timers;

            std::cout << E << "\t" << N << "\t";

            xt::xtensor<float, 1> ts = xt::empty<float>({N});

            generate_lorenz(ts, 40.0f / ts.size());

            auto train = xt::view(ts, xt::range(0, ts.size() / 2));
            auto test = xt::view(ts, xt::range(ts.size() / 2, ts.size()));

            for (int i = 0; i < N_WARMUPS + N_TRIALS; i++) {
                faiss::Index *index = nullptr;

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
