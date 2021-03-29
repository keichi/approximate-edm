#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include <omp.h>

#include "timer.hpp"

const size_t N = 100000;
const size_t K = 10;
const float DELTA = 0.001f;

struct Neighbor {
    size_t id;
    float distance;
    bool updated;

    friend bool operator<(const Neighbor &lhs, const Neighbor &rhs)
    {
        return lhs.distance < rhs.distance;
    }
};

struct NeighborList {
    std::vector<Neighbor> nodes;
    std::set<size_t> ids;
};

using NNGraph = std::vector<NeighborList>;

float sigma(const std::vector<float> &u, const std::vector<float> &v)
{
    float sum = 0.0f;

    assert(u.size() == 2 && v.size() == 2);

    for (size_t i = 0; i < u.size(); i++) {
        sum += (u[i] - v[i]) * (u[i] - v[i]);
    }

    return sum;
}

bool update_nns(NeighborList &nns, float l, size_t u, omp_lock_t *lock)
{
    if (l >= nns.nodes.front().distance) return false;
    if (nns.ids.find(u) != nns.ids.end()) return false;

    omp_set_lock(lock);

    nns.ids.erase(nns.nodes.front().id);
    nns.ids.insert(u);

    std::pop_heap(nns.nodes.begin(), nns.nodes.end());
    nns.nodes.back() = Neighbor{u, l, true};
    std::push_heap(nns.nodes.begin(), nns.nodes.end());

    omp_unset_lock(lock);

   return true;
}

void nn_descent(const std::vector<std::vector<float>> &data, NNGraph &nng)
{
    std::vector<omp_lock_t> locks(nng.size());

    for (auto &lock : locks) {
        omp_init_lock(&lock);
    }

    std::vector<std::set<size_t>> old_nns(nng.size()), new_nns(nng.size());


    Timer ttot;

    ttot.start();

    while (true) {
        Timer t1, t2;

        t1.start();

#pragma omp parallel for
        for (size_t v = 0; v < nng.size(); v++) {
            new_nns[v].clear();
            old_nns[v].clear();
        }

#pragma omp parallel for
        for (size_t v = 0; v < nng.size(); v++) {
            for (auto &node : nng[v].nodes) {
                if (node.updated) {
                    omp_set_lock(&locks[v]);
                    new_nns[v].insert(node.id);
                    omp_unset_lock(&locks[v]);

                    omp_set_lock(&locks[node.id]);
                    new_nns[node.id].insert(v);
                    omp_unset_lock(&locks[node.id]);
                } else {
                    omp_set_lock(&locks[v]);
                    old_nns[v].insert(node.id);
                    omp_unset_lock(&locks[v]);

                    omp_set_lock(&locks[node.id]);
                    old_nns[node.id].insert(v);
                    omp_unset_lock(&locks[node.id]);
                }

                node.updated = false;
            }
        }

        t1.stop();

        t2.start();

        auto c = 0;

#pragma omp parallel for
        for (size_t v = 0; v < nng.size(); v++) {
            for (const auto u1 : new_nns[v]) {
                for (const auto u2 : new_nns[v]) {
                    if (u1 >= u2) continue;

                    const auto l = sigma(data[u1], data[u2]);

                    if (update_nns(nng[u1], l, u2, &locks[u1])) c++;
                    if (update_nns(nng[u2], l, u1, &locks[u2])) c++;
                }

                for (const auto u2 : old_nns[v]) {
                    if (u1 == u2) continue;

                    const auto l = sigma(data[u1], data[u2]);

                    if (update_nns(nng[u1], l, u2, &locks[u1])) c++;
                    if (update_nns(nng[u2], l, u1, &locks[u2])) c++;
                }
            }
        }

        t2.stop();

        std::cerr << c << std::endl;
        std::cerr << "phase 1: " << t1.elapsed() << " [ms] " << std::endl;
        std::cerr << "phase 2: " << t2.elapsed() << " [ms]" << std::endl;

        if (c <= DELTA * N * K) break;
    }

    ttot.stop();

    std::cerr << "total: " << ttot.elapsed() << " [ms]" << std::endl;

    for (auto &lock : locks) {
        omp_destroy_lock(&lock);
    }
}

int main()
{
    // std::random_device rand_dev;
    // std::mt19937 engine(rand_dev());
    std::mt19937 engine;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> dist2(0, N - 1);

    std::vector<std::vector<float>> data(N);
    NNGraph nng(N);

    for (size_t i = 0; i < N; i++) {
        data[i].push_back(dist(engine));
        data[i].push_back(dist(engine));
        for (size_t j = 0; j < K; j++) {
            size_t id = dist2(engine);
            float distance = std::numeric_limits<float>::max();
            nng[i].nodes.push_back(Neighbor{id, distance, true});
            nng[i].ids.insert(id);
        }
    }

    nn_descent(data, nng);

    // std::cout << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
    //              "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"
    //           << std::endl;

    // for (size_t i = 0; i < N; i++) {
    //     std::cout << "<circle cx=\"" << (data[i][0] * 500) << "\" cy=\""
    //               << (data[i][1] * 500) << "\" r=\"3\" />" << std::endl;

    //     for (const auto id : nng[i].ids) {
    //         std::cout << "<line x1=\"" << (data[i][0] * 500) << "\" y1=\""
    //                   << (data[i][1] * 500) << "\" x2=\"" << (data[id][0] * 500)
    //                   << "\" y2=\"" << (data[id][1] * 500)
    //                   << "\" stroke=\"black\" />" << std::endl;
    //     }
    // }

    // std::cout << "</svg>" << std::endl;
}
