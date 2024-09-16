#include <iostream>
#include "MLP.hpp"

int main()
{
    // Define a 3-layer MLP: 3 inputs, then layers with 4, 4, and 1 neurons
    std::vector<int> nouts = {4, 4, 1};
    MLP mlp(3, nouts);

    // Example input
    std::vector<Value> input = {Value(0.5), Value(0.3), Value(0.2)};

    // Forward pass
    std::vector<Value> output = mlp(input);

    // Print the output
    std::cout << "MLP output: ";
    for (Value &v : output)
    {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    // Print the MLP structure
    std::cout << mlp << std::endl;

    return 0;
}