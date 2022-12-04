// #include <cstddef>
#include <iostream>
#include <vector>

#include <cxxopts.hpp>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_factory.h>
#include <xtensor.hpp>

#include "lorenz.hpp"

const int tau = 1;
const int Tp = 1;

void simplex(faiss::Index *index, const xt::xtensor<float, 1> &train,
             const xt::xtensor<float, 1> &test, int E)
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

    index->train(train_embed.shape(0), train_embed.data());
    index->add(train_embed.shape(0) - Tp, train_embed.data());
    index->search(test_embed.shape(0), test_embed.data(), E + 1, dist.data(),
                  ind.data());

    // Calculate weights from distances
    auto min_dist = xt::amin(dist, 1);
    auto w = xt::where(xt::expand_dims(min_dist > 0.0f, 1),
                       xt::exp(-dist / xt::expand_dims(min_dist, 1)),
                       xt::where(dist > 0.0f, 1.0f, 0.0f));
    auto w2 = xt::maximum(w, 1e-6f);
    auto w3 = w2 / xt::expand_dims(xt::sum(w2, 1), 1);

    xt::xtensor<float, 1> pred = xt::zeros<float>({ind.shape(0)});
    for (int i = 0; i < E + 1; i++) {
        pred += xt::index_view(train, xt::col(ind, i) + (E - 1) * tau + Tp) *
                xt::col(w3, i);
    }

    auto actual = xt::view(test, xt::range((E - 1) * tau + Tp, test.size()));
    auto predicted = xt::view(pred, xt::range(0, pred.size() - 1));

    float mape = xt::mean(xt::abs((actual - predicted) / actual))(0);
    std::cout << std::fixed << std::setprecision(5) << mape
              << std::endl;
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("eval-knn-recall",
                             "Compare recall of AkNN algorithms");

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
    std::cout << "E\tN\tMAPE" << std::endl;

    for (int E : result["embedding-dims"].as<std::vector<int>>()) {
        int start = result["log2-min-n"].as<int>();
        int end = result["log2-max-n"].as<int>();

        for (int N = 1 << start; N <= 1 << end; N <<= 1) {
            std::cout << E << "\t" << N << "\t";

            xt::xtensor<float, 1> ts = xt::empty<float>({N});

            generate_lorenz(ts, 40.0f / ts.size());

            auto train = xt::view(ts, xt::range(0, ts.size() / 2));
            auto test = xt::view(ts, xt::range(ts.size() / 2, ts.size()));

            faiss::Index *index = faiss::index_factory(
                E, result["index"].as<std::string>().c_str());

            if (result.count("gpu")) {
                index = faiss::gpu::index_cpu_to_gpu(&res, 0, index);
            }

            simplex(index, train, test, E);
        }
    }

    return 0;
}
