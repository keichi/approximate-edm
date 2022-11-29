#include <cstddef>
#include <iostream>
#include <vector>

#include <xtensor/xindex_view.hpp>
#include <xtensor/xview.hpp>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

#include "timer.hpp"

const int tau = 1;
const int Tp = 1;

const float p = 10.0f;
const float r = 28.0f;
const float b = 8.0f / 3.0f;

float f(float t, float x, float y, float z)
{
    return (p * y - p * x); // dx/dt = -px + py
}

float g(float t, float x, float y, float z)
{
    return (r * x - x * z - y); // dy/dt = -xz + rx - y
}

float h(float t, float x, float y, float z)
{
    return (x * y - b * z); // dz/dt = xy -bz
}

void generate_dataset(xt::xtensor<float, 1> &ts, float DT)
{
    float x = 0.1f, y = 0.1f, z = 0.1f, t = 0.0f;
    float k1, k2, k3, k4;
    float l1, l2, l3, l4;
    float m1, m2, m3, m4;

    for (int i = 0; i < ts.size(); i++) {
        k1 = DT * f(t, x, y, z);
        l1 = DT * g(t, x, y, z);
        m1 = DT * h(t, x, y, z);
        k2 = DT * f(t + DT / 2.0f, x + k1 / 2.0f, y + l1 / 2.0f, z + m1 / 2.0f);
        l2 = DT * g(t + DT / 2.0f, x + k1 / 2.0f, y + l1 / 2.0f, z + m1 / 2.0f);
        m2 = DT * h(t + DT / 2.0f, x + k1 / 2.0f, y + l1 / 2.0f, z + m1 / 2.0f);
        k3 = DT * f(t + DT / 2.0f, x + k2 / 2.0f, y + l2 / 2.0f, z + m2 / 2.0f);
        l3 = DT * g(t + DT / 2.0f, x + k2 / 2.0f, y + l2 / 2.0f, z + m2 / 2.0f);
        m3 = DT * h(t + DT / 2.0f, x + k2 / 2.0f, y + l2 / 2.0f, z + m2 / 2.0f);
        k4 = DT * f(t + DT, x + k3, y + l3, z + m3);
        l4 = DT * g(t + DT, x + k3, y + l3, z + m3);
        m4 = DT * h(t + DT, x + k3, y + l3, z + m3);
        x += (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0F;
        y += (l1 + 2.0f * l2 + 2.0f * l3 + l4) / 6.0F;
        z += (m1 + 2.0f * m2 + 2.0f * m3 + m4) / 6.0F;
        t += DT;

        ts(i) = x;
    }
}

void simplex(faiss::Index &index, const xt::xtensor<float, 1> &train,
             const xt::xtensor<float, 1> &test, int E)
{
    xt::xtensor<float, 2> train_embed = xt::zeros<float>(
        {train.shape(0) - (E - 1) * tau, static_cast<size_t>(E)});
    xt::xtensor<float, 2> test_embed = xt::zeros<float>(
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

    Timer timer_train, timer_index, timer_search, timer_total;

    timer_total.start();

    timer_train.start();
    index.train(train_embed.shape(0), train_embed.data());
    timer_train.stop();
    std::cout << "Train index: " << timer_train.elapsed() << " [ms]"
              << std::endl;

    timer_index.start();
    index.add(train_embed.shape(0) - Tp, train_embed.data());
    timer_index.stop();
    std::cout << "Build index: " << timer_index.elapsed() << " [ms]"
              << std::endl;

    timer_search.start();
    index.search(test_embed.shape(0), test_embed.data(), E + 1, dist.data(),
                 ind.data());
    timer_search.stop();
    timer_total.stop();
    std::cout << "Search kNN: " << timer_search.elapsed() << " [ms]"
              << std::endl;
    std::cout << "Total: " << timer_total.elapsed() << " [ms]" << std::endl;

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
    std::cout << "MAPE: " << mape << std::endl;

    // Calculate recall of NNs
    {
        faiss::gpu::StandardGpuResources res;
        faiss::gpu::GpuIndexFlatL2 index(&res, E);

        xt::xtensor<float, 2> dist_exact =
            xt::empty<float>({test_embed.shape(0), static_cast<size_t>(E + 1)});
        xt::xtensor<faiss::Index::idx_t, 2> ind_exact =
            xt::empty<faiss::Index::idx_t>(
                {test_embed.shape(0), static_cast<size_t>(E + 1)});

        index.train(train_embed.shape(0), train_embed.data());
        index.add(train_embed.shape(0) - Tp, train_embed.data());
        index.search(test_embed.shape(0), test_embed.data(), E + 1,
                     dist_exact.data(), ind_exact.data());

        size_t correct = 0;
        #pragma omp parallel for reduction(+:correct)
        for (int i = 0; i < ind_exact.shape(0); i++) {
            for (int j = 0; j < ind_exact.shape(1); j++) {
                faiss::Index::idx_t idx = ind_exact(i, j);

                for (int k = 0; k < ind_exact.shape(1); k++) {
                    if (ind(i, k) == ind_exact(i, j)) {
                        correct++;
                        break;
                    }
                }
            }
        }

        float recall = static_cast<float>(correct) / ind_exact.size();

        std::cout << "Recall: " << recall << std::endl;
    }

}

int main()
{
    for (int N = 1 << 10; N <= 1 << 20; N <<= 1) {
        std::cout << "N=" << N << std::endl;

        xt::xtensor<float, 1> ts = xt::zeros<float>({N});

        generate_dataset(ts, 40.0f / ts.size());

        const auto train = xt::view(ts, xt::range(0, ts.size() / 2));
        const auto test = xt::view(ts, xt::range(ts.size() / 2, ts.size()));

        int E = 20;

        // HNSW M=k
        // M: number of connections that would be made for each new vertex
        // during construction efConstruction: number of candidate neighbors to
        // explore during construction time efSearch: number of candidate
        // neighbors to explore during search time
        faiss::IndexHNSWFlat index(E, E + 1);
        index.hnsw.efConstruction = E + 1;
        index.hnsw.efSearch = E + 1;

        simplex(index, train, test, E);
    }

    return 0;
}
