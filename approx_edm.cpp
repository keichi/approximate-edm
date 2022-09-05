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
#include <faiss/IndexLSH.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <nanoflann.hpp>

#include "timer.hpp"

const int N = 1000000;
const float DT = 40.0f / N;
const int E = 5;
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
    xt::xtensor<faiss::Index::idx_t, 2> ind = xt::empty<faiss::Index::idx_t>(
        {test_embed.shape(0), static_cast<size_t>(E + 1)});
    // xt::xtensor<unsigned int, 2> ind = xt::empty<faiss::Index::idx_t>(
        // {test_embed.shape(0), static_cast<size_t>(E + 1)});

    faiss::gpu::StandardGpuResources res; // use a single GPU

    Timer timer_train, timer_index, timer_search, timer_total;

    // Brute-force on CPU -> 462823 ms
    // faiss::IndexFlatL2 index(E);

    // Brute-force on GPU -> 7297.32 ms
    faiss::gpu::GpuIndexFlatL2 index(&res, E);

    // Inverted file on CPU -> 4000.96 ms
    // faiss::IndexFlatL2 quantizer(E);
    // faiss::IndexIVFFlat index(&quantizer, E, 256);

    // HNSW on CPU -> 801.73 ms
    // faiss::IndexHNSWFlat index(E, 4);
    // index.hnsw.efConstruction = 2;
    // index.hnsw.efSearch = 2;

    // NSG on CPU -> ???
    // faiss::IndexNSGFlat index(E, 32);

    // Inverted file + HNSQ on CPU -> 8719.06 ms
    // faiss::IndexHNSWFlat quantizer(E, 4);
    // quantizer.hnsw.efConstruction = 2;
    // quantizer.hnsw.efSearch = 2;
    // faiss::IndexIVFFlat index(&quantizer, E, 256);

    // PQ on CPU -> ???
    // faiss::IndexPQ index(E, 1, 8);

    // Inverted file + PQ on CPU -> 2216.23 ms
    // faiss::IndexFlatL2 quantizer(E);
    // faiss::IndexIVFPQ index(&quantizer, E, 256, 1, 4);

    // NN-Descent on CPU -> 73088.4 ms, R2: 0.291908
    // faiss::IndexNNDescentFlat index(E, E+1);

    // LSH -> incorrect result
    // faiss::IndexLSH index(E, 8);

    timer_total.start();

    timer_train.start();
    index.train(train_embed.shape(0), train_embed.data());
    timer_train.stop();
    std::cout << "Train index: " << timer_train.elapsed() << " [ms]" << std::endl;

    timer_index.start();
    index.add(train_embed.shape(0) - Tp, train_embed.data());
    timer_index.stop();
    std::cout << "Build index: " << timer_index.elapsed() << " [ms]" << std::endl;

    timer_search.start();
    index.search(test_embed.shape(0), test_embed.data(), E + 1, dist.data(),
                 ind.data());
    timer_search.stop();
    timer_total.stop();
    std::cout << "Search kNN: " << timer_search.elapsed() << " [ms]" << std::endl;

    // timer_total.start();
    // timer_index.start();

    // k-d tree on CPU -> 13417.4 ms
    // XtensorDataset<float> dataset(train_embed);
    // nanoflann::KDTreeSingleIndexAdaptor<
        // nanoflann::L2_Simple_Adaptor<float, XtensorDataset<float>>,
        // XtensorDataset<float>>
        // index(E, dataset);
    // index.buildIndex();
    // timer_index.stop();
    // std::cout << "Build index: " << timer_index.elapsed() << " [ms]" << std::endl;

    // timer_search.start();
    // #pragma omp parallel for
    // for (int i = 0; i < test_embed.shape(0); i++) {
        // index.knnSearch(xt::row(test_embed, i).data() + xt::row(test_embed, i).data_offset(),
                        // E+1,
                        // xt::row(ind, i).data() + xt::row(ind, i).data_offset(),
                        // xt::row(dist, i).data() + xt::row(dist, i).data_offset());
    // }

    // timer_search.stop();
    // std::cout << "Search kNN: " << timer_search.elapsed() << " [ms]" << std::endl;
    // timer_total.stop();

    std::cout << "Total: " << timer_total.elapsed() << " [ms]" << std::endl;

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

    // dist = xt::broadcast(0.1f, dist.shape());

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

    float actual_mean = xt::mean(actual)(0);
    float r2 = 1.0f - xt::sum(xt::square(actual - predicted))(0) /
                          xt::sum(xt::square(actual - actual_mean))(0);
    std::cout << "R2: " << r2 << std::endl;

    float mape = xt::mean(xt::abs((actual - predicted) / actual))(0);
    std::cout << "MAPE: " << mape << std::endl;

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
