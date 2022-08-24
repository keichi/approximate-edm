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
#include <tbb/concurrent_unordered_set.h>

#include "timer.hpp"

const size_t N = 1000000;
const size_t K = 20;
const float DELTA = 0.001f;
const float MAX_ITER = 3;

struct Neighbor {
    size_t id;
    float distance;
    bool updated;

    friend bool operator<(const Neighbor &lhs, const Neighbor &rhs)
    {
        return lhs.distance < rhs.distance;
    }
};

struct NNGraph {
    int n;
    std::vector<Neighbor> nns;

    NNGraph(size_t n) : n(n), nns(K * n) {}

    size_t size() const {
        return n;
    }

    // v: this node, u: potential neighbor, l: distance between u and v
    bool insert_neighbor(size_t v, float l, size_t u)
    {
        if (l >= nns[K * v].distance) {
            return false;
        }

        for (size_t i = K * v; i < K * (v + 1); i++) {
            if (u == nns[i].id) {
                return false;
            }
        }

        std::pop_heap(nns.begin() + K * v, nns.begin() + K * (v + 1));
        nns[K * (v + 1) - 1] = Neighbor{u, l, true};
        std::push_heap(nns.begin() + K * v, nns.begin() + K * (v + 1));

        return true;
    }

    const Neighbor& kth_neighbor(size_t v, size_t k) const {
        return nns[K * v + k];
    }

    Neighbor& kth_neighbor(size_t v, size_t k) {
        return nns[K * v + k];
    }
};

float sigma(const std::vector<float> &u, const std::vector<float> &v)
{
    float sum = 0.0f;

    assert(u.size() == 2 && v.size() == 2);

    for (size_t i = 0; i < u.size(); i++) {
        sum += (u[i] - v[i]) * (u[i] - v[i]);
    }

    return sum;
}

void nn_descent(const std::vector<std::vector<float>> &data, NNGraph &nng)
{
    std::vector<NNGraph> local_nng(omp_get_max_threads(), nng.size());
    std::vector<tbb::concurrent_unordered_set<size_t>> old_nns(nng.size()),
        new_nns(nng.size());

    Timer timer_total, timer_phase0_total, timer_phase1_total,
          timer_phase2_total, timer_phase3_total;

    timer_total.start();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        Timer timer_phase0, timer_phase1, timer_phase2, timer_phase3;

        timer_phase0_total.start();
        timer_phase0.start();

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            local_nng[tid] = nng;
        }

        timer_phase0.stop();
        timer_phase0_total.stop();

        timer_phase1_total.start();
        timer_phase1.start();

#pragma omp parallel for
        for (size_t v = 0; v < nng.size(); v++) {
            new_nns[v].clear();
            old_nns[v].clear();
        }

#pragma omp parallel for
        for (size_t v = 0; v < nng.size(); v++) {
            for (size_t k = 0; k < K; k++) {
                Neighbor& node = nng.kth_neighbor(v, k);

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

#pragma omp parallel
        {
            int tid = omp_get_thread_num();

#pragma omp for
            for (size_t v = 0; v < nng.size(); v++) {
                for (const auto u1 : new_nns[v]) {
                    // float min_dist = std::numeric_limits<float>::max();
                    // size_t min_id = 0;

                    for (const auto u2 : new_nns[v]) {
                        if (u1 >= u2) continue;

                        float dist = sigma(data[u1], data[u2]);

                        local_nng[tid].insert_neighbor(u1, dist, u2);
                        local_nng[tid].insert_neighbor(u2, dist, u1);

                        // if (dist < min_dist) {
                            // min_dist = dist;
                            // min_id = u2;
                        // }
                    }

                    // local_nng[tid].insert_neighbor(u1, min_dist, min_id);
                    // local_nng[tid].insert_neighbor(min_id, min_dist, u1);
                }

                for (const auto u1 : new_nns[v]) {
                    // float min_dist = std::numeric_limits<float>::max();
                    // size_t min_id = 0;

                    for (const auto u2 : old_nns[v]) {
                        if (u1 == u2) continue;

                        float dist = sigma(data[u1], data[u2]);

                        local_nng[tid].insert_neighbor(u1, dist, u2);
                        local_nng[tid].insert_neighbor(u2, dist, u1);

                        // if (dist < min_dist) {
                            // min_dist = dist;
                            // min_id = u2;
                        // }
                    }

                    // local_nng[tid].insert_neighbor(u1, min_dist, min_id);
                    // local_nng[tid].insert_neighbor(min_id, min_dist, u1);
                }
            }
        }

        timer_phase2.stop();
        timer_phase2_total.stop();

        int c = 0;

        timer_phase3_total.start();
        timer_phase3.start();
#pragma omp parallel for reduction(+:c)
        for (size_t v = 0; v < nng.size(); v++) {
            for (size_t tid = 0; tid < local_nng.size(); tid++) {
                for (size_t k = 0; k < K; k++) {
                    const auto& new_nn = local_nng[tid].kth_neighbor(v, k);
                    if (nng.insert_neighbor(v, new_nn.distance, new_nn.id)) {
                        c++;
                    }
                }
            }
        }
        timer_phase3.stop();
        timer_phase3_total.stop();

        std::cerr << "Iteration #" << iter << ": updated " << c << " neighbors"
                  << std::endl;
        std::cerr << "\tPhase 0: " << timer_phase0.elapsed() << " [ms], "
                  << "Phase 1: " << timer_phase1.elapsed() << " [ms], "
                  << "Phase 2: " << timer_phase2.elapsed() << " [ms], "
                  << "Phase 3: " << timer_phase3.elapsed() << " [ms]"
                  << std::endl;

        if (c <= DELTA * N * K) break;
    }

    timer_total.stop();

    std::cerr << "Total: " << timer_total.elapsed() << " [ms]" << std::endl;
    std::cerr << "\tPhase 0: " << timer_phase0_total.elapsed() << " [ms], "
              << "Phase 1: " << timer_phase1_total.elapsed() << " [ms], "
              << "Phase 2: " << timer_phase2_total.elapsed() << " [ms], "
              << "Phase 3: " << timer_phase3_total.elapsed() << " [ms]"
              << std::endl;
}

void bruteforce(const std::vector<std::vector<float>> &data, NNGraph &nng)
{
    Timer timer_total;

    timer_total.start();

#pragma omp parallel
    {
        std::vector<float> distances(data.size());
        std::vector<size_t> indices(data.size());

#pragma omp for
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

            for (size_t k = 0; k < K; k++) {
                auto &node = nng.kth_neighbor(i, k);
                node.id = indices[k];
                node.distance = distances[k];
            }
        }
    }

    timer_total.stop();

    std::cerr << "Total: " << timer_total.elapsed() << " [ms]" << std::endl;
}

void eval_nng(const std::vector<std::vector<float>> &data, const NNGraph &nng)
{
    std::vector<float> distances(data.size());
    std::vector<size_t> indices(data.size());

    size_t tp = 0;

#pragma omp parallel
    {
        std::vector<float> distances(data.size());
        std::vector<size_t> indices(data.size());
        std::unordered_set<size_t> set;

#pragma omp for reduction(+:tp)
        for (size_t i = 0; i < data.size(); i++) {
            std::fill(distances.begin(), distances.end(), 0.0f);
            std::iota(indices.begin(), indices.end(), 0);
            set.clear();

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

            for (size_t k = 0; k < K; k++) {
                set.insert(nng.kth_neighbor(i, k).id);
            }

            for (size_t k = 0; k < K; k++) {
                if (set.find(indices[k]) != set.end()) {
                    tp++;
                }
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
        for (size_t k = 0; k < K; k++) {
            auto& node = nng.kth_neighbor(i, k);

            node.id = dist2(engine);
            node.distance = std::numeric_limits<float>::max();
            node.updated = true;
        }
    }

    std::cerr << "NN-Descent" << std::endl;
    nn_descent(data, nng);

    eval_nng(data, nng);

    std::cerr << "Bruteforce" << std::endl;
    bruteforce(data, nng);

    eval_nng(data, nng);
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
