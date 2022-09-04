#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexNNDescent.h>

#include "timer.hpp"

const int N = 100000;
const float DT = 40.0f / N;
const int E = 3;
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

void generate_dataset(xt::xtensor<float, 1> &ts)
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

void simplex(const xt::xtensor<float, 1> &train,
             const xt::xtensor<float, 1> &test)
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
    xt::xtensor<faiss::Index::idx_t, 2> ind =
        xt::empty<faiss::Index::idx_t>({test_embed.shape(0), static_cast<size_t>(E + 1)});

    Timer timer;

    timer.start();

    // faiss::IndexFlatL2 index(E);
    // faiss::IndexHNSWFlat index(E, 4);
    // faiss::IndexPQ index(E, 1, 8);
    faiss::IndexNNDescentFlat index(E, E+1);
    // index.train(train_embed.shape(0) - Tp, train_embed.data());
    index.add(train_embed.shape(0) - Tp, train_embed.data());
    index.search(test_embed.shape(0), test_embed.data(), E+1, dist.data(), ind.data());

    timer.stop();
    std::cout << "kNN search: " << timer.elapsed() << "[ms]" << std::endl;

    // // Calculate pairwise distance matrix
    // xt::xtensor<float, 2> dmatrix =
    //     xt::sum(xt::square(xt::expand_dims(test_embed, 1) -
    //                        xt::expand_dims(
    //                            xt::view(train_embed,
    //                                     xt::range(0, train_embed.shape(0) -
    //                                     Tp), xt::all()),
    //                            0)),
    //             2);

    // // Find k-nearest neighbors
    // auto ind =
    //     xt::view(xt::argsort(dmatrix, 1), xt::all(), xt::range(0, E + 1));
    // auto dist = xt::sqrt(
    //     xt::view(xt::sort(dmatrix, 1), xt::all(), xt::range(0, E + 1)));

    // Calculate weights from distances
    auto w = xt::exp(-dist / xt::expand_dims(xt::amin(dist, 1), 1));
    auto w2 = w / xt::expand_dims(xt::sum(w, 1), 1);

    xt::xtensor<float, 1> pred = xt::zeros<float>({ind.shape(0)});

    for (int i = 0; i < E + 1; i++) {
        pred += xt::index_view(train, xt::col(ind, i) + (E - 1) * tau + Tp) *
                xt::col(w2, i);
    }

    auto actual = xt::view(test, xt::range((E - 1) * tau + Tp, test.size()));
    auto predicted = xt::view(pred, xt::range(0, pred.size() - 1));

    float r2 = 1.0f - xt::sum(xt::square(actual - predicted))(0) /
                          xt::sum(xt::square(actual - xt::mean(actual)))(0);

    std::cout << "R2: " << r2 << std::endl;

    std::ofstream f1("pred.csv");
    xt::dump_csv(f1, xt::expand_dims(pred, 1));

    std::ofstream f2("test.csv");
    xt::dump_csv(f2, xt::expand_dims(test, 1));
}

int main()
{
    xt::xtensor<float, 1> ts = xt::zeros<float>({N});

    generate_dataset(ts);

    const auto train = xt::view(ts, xt::range(0, ts.size() / 2));
    const auto test = xt::view(ts, xt::range(ts.size() / 2, ts.size()));

    simplex(train, test);

    return 0;
}
