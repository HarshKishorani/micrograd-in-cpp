#include "neuron.hpp"

Neuron::Neuron(int nin, bool nonlin)
{
    for (int i = 0; i < nin; i++)
    {
        this->w.emplace_back(Value(RandomUtils::getRandomUniform()));
    }
    this->b = Value(0);
    this->nonlin = nonlin;
}

Value Neuron::operator()(const std::vector<Value> &x)
{
    if (x.size() != this->w.size())
    {
        throw std::invalid_argument("Input size does not match the number of neuron weights");
    }

    Value sum(0);

    for (int i = 0; i < this->w.size(); i++)
    {
        Value &wi = this->w[i];
        Value xi = x[i];
        sum = sum + (wi * xi); 
    }

    sum = sum + this->b;

    if (this->nonlin)
    {
        return sum.relu();
    }

    return sum;
}

std::vector<Value *> Neuron::parameters()
{
    std::vector<Value *> params;

    for (Value &value : this->w)
    {
        params.emplace_back(&value);
    }

    params.emplace_back(&this->b);

    return params;
}

std::ostream &operator<<(std::ostream &os, Neuron const &neuron)
{
    return os << std::format("{}Neuron({})\n", neuron.nonlin ? "Relu" : "Linear", neuron.w.size());
}
