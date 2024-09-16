#include "MLP.hpp"

MLP::MLP(int nin, const std::vector<int> &nouts)
{
    // Create layers: input size for the first layer is nin, then subsequent layers take nouts[i]
    int input_size = nin;
    for (int nout : nouts)
    {
        this->layers.emplace_back(Layer(input_size, nout)); // Create Layer with nin and nout
        input_size = nout;                                  // Next layer's input size is this layer's output size
    }
}

std::vector<Value> MLP::operator()(std::vector<Value> &x)
{
    for (Layer &layer : this->layers)
    {
        x = layer(x); 
    }
    return x;
}

std::vector<Value *> MLP::parameters()
{
    std::vector<Value *> params;
    // Collect parameters from all layers in the MLP
    for (Layer &layer : this->layers)
    {
        std::vector<Value *> layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

// Output representation of the MLP using << operator
std::ostream &operator<<(std::ostream &os, const MLP &mlp)
{
    os << "MLP of [";
    for (size_t i = 0; i < mlp.layers.size(); ++i)
    {
        os << mlp.layers[i]; // Assuming operator<< is implemented for Layer
        if (i < mlp.layers.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
