#include <cassert>
#include <iostream>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

const size_t N = 1000;
const size_t K = 10;

struct Neighbor {
    size_t id;
    float distance;

    friend bool operator<(const Neighbor &lhs, const Neighbor &rhs)
    {
        return lhs.distance < rhs.distance;
    }
};

struct NeighborList {
    std::priority_queue<Neighbor> nodes;
    std::unordered_set<size_t> ids;
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

void nn_descent(const std::vector<std::vector<float>> &data, NNGraph &nng)
{
    while (true) {
        std::vector<std::unordered_set<size_t>> neighbors(N);

        for (size_t v = 0; v < nng.size(); v++) {
            for (const auto u : nng[v].ids) {
                neighbors[v].insert(u);
                neighbors[u].insert(v);
            }
        }

        auto c = 0;

        for (size_t v = 0; v < nng.size(); v++) {
            for (const auto u1 : neighbors[v]) {
                for (const auto u2 : neighbors[u1]) {
                    if (v == u2) continue;

                    if (nng[v].ids.find(u2) != nng[v].ids.end()) continue;

                    const auto l = sigma(data[v], data[u2]);

                    const auto top = nng[v].nodes.top();
                    if (l >= top.distance) continue;

                    nng[v].nodes.pop();
                    nng[v].nodes.push(Neighbor{u2, l});

                    nng[v].ids.erase(top.id);
                    nng[v].ids.insert(u2);

                    c += 1;
                }
            }
        }

        std::cerr << c << std::endl;

        if (c == 0) break;
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
            nng[i].nodes.push(Neighbor{id, distance});
            nng[i].ids.insert(id);
        }
    }

    nn_descent(data, nng);

    std::cout << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
                 "xmlns:xlink=\"http://www.w3.org/1999/xlink\">"
              << std::endl;

    for (size_t i = 0; i < N; i++) {
        std::cout << "<circle cx=\"" << (data[i][0] * 500) << "\" cy=\""
                  << (data[i][1] * 500) << "\" r=\"3\" />" << std::endl;

        for (const auto id : nng[i].ids) {
            std::cout << "<line x1=\"" << (data[i][0] * 500) << "\" y1=\""
                      << (data[i][1] * 500) << "\" x2=\"" << (data[id][0] * 500)
                      << "\" y2=\"" << (data[id][1] * 500)
                      << "\" stroke=\"black\" />" << std::endl;
        }
    }

    std::cout << "</svg>" << std::endl;
}
