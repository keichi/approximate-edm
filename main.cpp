// g++ -Wall -fopenmp -O3 -g main.cpp -ltbb

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include <omp.h>
#include <pthread.h>
#include <tbb/concurrent_unordered_set.h>

#include "timer.hpp"

const size_t N = 1000000;
const size_t K = 20;
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

using NeighborList = std::vector<Neighbor>;
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

bool update_nns(NeighborList &nns, float l, size_t u, pthread_rwlock_t *lock)
{
    pthread_rwlock_rdlock(lock);

    if (l >= nns.front().distance) {
        pthread_rwlock_unlock(lock);
        return false;
    }

    for (const auto &v : nns) {
        if (u == v.id) {
            pthread_rwlock_unlock(lock);
            return false;
        }
    }

    pthread_rwlock_unlock(lock);

    pthread_rwlock_wrlock(lock);

    std::pop_heap(nns.begin(), nns.end());
    nns.back() = Neighbor{u, l, true};
    std::push_heap(nns.begin(), nns.end());

    pthread_rwlock_unlock(lock);

    return true;
}

void nn_descent(const std::vector<std::vector<float>> &data, NNGraph &nng)
{
    std::vector<pthread_rwlock_t> locks(nng.size(), PTHREAD_RWLOCK_INITIALIZER);
    std::vector<tbb::concurrent_unordered_set<size_t>> old_nns(nng.size()),
        new_nns(nng.size());

    Timer timer_total, timer_phase1_total, timer_phase2_total;

    timer_total.start();

    for (int iter = 0;; iter++) {
        Timer timer_phase1, timer_phase2;

        timer_phase1_total.start();
        timer_phase1.start();

#pragma omp parallel for
        for (size_t v = 0; v < nng.size(); v++) {
            new_nns[v].clear();
            old_nns[v].clear();
        }

#pragma omp parallel for
        for (size_t v = 0; v < nng.size(); v++) {
            for (auto &node : nng[v]) {
                if (node.updated) {
                    new_nns[v].insert(node.id);
                    new_nns[node.id].insert(v);
                } else {
                    old_nns[v].insert(node.id);
                    old_nns[node.id].insert(v);
                }

                node.updated = false;
            }
        }

        timer_phase1.stop();
        timer_phase1_total.stop();

        timer_phase2_total.start();
        timer_phase2.start();

        auto c = 0;

#pragma omp parallel for reduction(+:c)
        for (size_t v = 0; v < nng.size(); v++) {
            for (const auto u1 : new_nns[v]) {
                float min_dist = std::numeric_limits<float>::max();
                size_t min_id = 0;

                for (const auto u2 : new_nns[v]) {
                    if (u1 >= u2) continue;

                    float dist = sigma(data[u1], data[u2]);

                    if (dist < min_dist) {
                        min_dist = dist;
                        min_id = u2;
                    }
                }

                if (update_nns(nng[u1], min_dist, min_id, &locks[u1])) {
                    c++;
                }
                if (update_nns(nng[min_id], min_dist, u1, &locks[min_id])) {
                    c++;
                }
            }

            for (const auto u1 : new_nns[v]) {
                float min_dist = std::numeric_limits<float>::max();
                size_t min_id = 0;

                for (const auto u2 : old_nns[v]) {
                    if (u1 == u2) continue;

                    float dist = sigma(data[u1], data[u2]);

                    if (dist < min_dist) {
                        min_dist = dist;
                        min_id = u2;
                    }
                }

                if (update_nns(nng[u1], min_dist, min_id, &locks[u1])) {
                    c++;
                }
                if (update_nns(nng[min_id], min_dist, u1, &locks[min_id])) {
                    c++;
                }
            }
        }

        timer_phase2.stop();
        timer_phase2_total.stop();

        // std::cerr << "Iteration #" << iter << ": updated " << c << " neighbors"
        //           << std::endl;
        // std::cerr << "\tPhase 1: " << timer_phase1.elapsed() << " [ms], "
        //           << "Phase 2: " << timer_phase2.elapsed() << " [ms]"
        //           << std::endl;

        if (c <= DELTA * N * K) break;
    }

    timer_total.stop();

    std::cerr << "Total: " << timer_total.elapsed() << " [ms]" << std::endl;
    std::cerr << "\tPhase 1: " << timer_phase1_total.elapsed() << " [ms], "
              << "Phase 2: " << timer_phase2_total.elapsed() << " [ms]"
              << std::endl;

    for (auto &lock : locks) {
        pthread_rwlock_destroy(&lock);
    }
}

void eval_nng(const std::vector<std::vector<float>> &data, const NNGraph &nng)
{
    std::vector<float> distances(data.size());
    std::vector<size_t> indices(data.size());

    size_t tp = 0;

    for (size_t i = 0; i < data.size(); i++) {
        std::fill(distances.begin(), distances.end(), 0.0f);
        std::iota(indices.begin(), indices.end(), 0);

        distances[i] = std::numeric_limits<float>::max();

        for (size_t j = 0; j < data.size(); j++) {
            for (size_t k = 0; k < data[j].size(); k++) {
                float diff = data[i][k] - data[j][k];
                distances[j] += diff * diff;
            }
        }

        std::partial_sort(
            indices.begin(), indices.begin() + K, indices.end(),
            [&](size_t a, size_t b) { return distances[a] < distances[b]; });

        std::unordered_set<size_t> set;
        for (size_t k = 0; k < K; k++) {
            set.insert(nng[i][k].id);
        }

        for (size_t k = 0; k < K; k++) {
            if (set.find(indices[k]) != set.end()) {
                tp++;
            }
        }
    }

    std::cerr << "Recall: " << std::fixed << std::setprecision(3)
              << (static_cast<float>(tp) / (K * data.size())) << std::endl;
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

        // Randomly initialize neighbors
        for (size_t j = 0; j < K; j++) {
            size_t id = dist2(engine);
            float distance = std::numeric_limits<float>::max();
            nng[i].push_back(Neighbor{id, distance, true});
        }
    }

    nn_descent(data, nng);
    // eval_nng(data, nng);

    // std::cout << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
    // "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"
    // << std::endl;

    // for (size_t i = 0; i < N; i++) {
    // std::cout << "<circle cx=\"" << (data[i][0] * 500) << "\" cy=\""
    // << (data[i][1] * 500) << "\" r=\"3\" />" << std::endl;

    // for (const auto id : nng[i].ids) {
    // std::cout << "<line x1=\"" << (data[i][0] * 500) << "\" y1=\""
    // << (data[i][1] * 500) << "\" x2=\"" << (data[id][0] * 500)
    // << "\" y2=\"" << (data[id][1] * 500)
    // << "\" stroke=\"black\" />" << std::endl;
    // }
    // }

    // std::cout << "</svg>" << std::endl;
}
