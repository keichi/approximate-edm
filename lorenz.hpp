#pragma once

#include <xtensor.hpp>

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

void generate_lorenz(xt::xtensor<float, 1> &ts, float dt)
{
    float x = 0.1f, y = 0.1f, z = 0.1f, t = 0.0f;
    float k1, k2, k3, k4;
    float l1, l2, l3, l4;
    float m1, m2, m3, m4;

    for (int i = 0; i < ts.size(); i++) {
        k1 = dt * f(t, x, y, z);
        l1 = dt * g(t, x, y, z);
        m1 = dt * h(t, x, y, z);
        k2 = dt * f(t + dt / 2.0f, x + k1 / 2.0f, y + l1 / 2.0f, z + m1 / 2.0f);
        l2 = dt * g(t + dt / 2.0f, x + k1 / 2.0f, y + l1 / 2.0f, z + m1 / 2.0f);
        m2 = dt * h(t + dt / 2.0f, x + k1 / 2.0f, y + l1 / 2.0f, z + m1 / 2.0f);
        k3 = dt * f(t + dt / 2.0f, x + k2 / 2.0f, y + l2 / 2.0f, z + m2 / 2.0f);
        l3 = dt * g(t + dt / 2.0f, x + k2 / 2.0f, y + l2 / 2.0f, z + m2 / 2.0f);
        m3 = dt * h(t + dt / 2.0f, x + k2 / 2.0f, y + l2 / 2.0f, z + m2 / 2.0f);
        k4 = dt * f(t + dt, x + k3, y + l3, z + m3);
        l4 = dt * g(t + dt, x + k3, y + l3, z + m3);
        m4 = dt * h(t + dt, x + k3, y + l3, z + m3);
        x += (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0F;
        y += (l1 + 2.0f * l2 + 2.0f * l3 + l4) / 6.0F;
        z += (m1 + 2.0f * m2 + 2.0f * m3 + m4) / 6.0F;
        t += dt;

        ts(i) = x;
    }
}
