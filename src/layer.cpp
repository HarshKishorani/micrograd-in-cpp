#include "layer.hpp"

Layer::Layer(int nin, int nout, bool nonlin)
{
    for (int i = 0; i < nout; i++)
    {
        this->neurons.emplace_back(Neuron(nin, nonlin));
    }
}

std::vector<Value> Layer::operator()(const std::vector<Value> &x)
{
    std::vector<Value> out;
    // Forward pass through each neuron in the layer
    for (Neuron &n : this->neurons)
    {
        out.emplace_back(n(x));
    }
    return out;
}

std::vector<Value *> Layer::parameters()
{
    std::vector<Value *> params;
    // Collect parameters from all neurons in the layer
    for (Neuron &n : this->neurons)
    {
        std::vector<Value *> neuron_params = n.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

// Output representation of the layer using << operator
std::ostream &operator<<(std::ostream &os, const Layer &layer)
{
    os << "Layer of [\n";
    for (size_t i = 0; i < layer.neurons.size(); ++i)
    {
        os << layer.neurons[i];
    }
    os << "]";
    return os;
}