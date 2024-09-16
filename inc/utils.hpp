#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>

class RandomUtils
{
public:
    // Static function to generate a random floating-point number between -1 and 1
    static double getRandomUniform()
    {
        static std::mt19937 gen(std::random_device{}()); // static to initialize once
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        return dis(gen);
    }
};

#endif // RANDOM_UTILS_H
