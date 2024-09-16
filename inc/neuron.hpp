#ifndef NEURON_H
#define NEURON_H

#include "module.hpp"
#include "value.hpp"
#include "utils.hpp"

class Neuron : public Module
{
private:
    std::vector<Value> w; // weights
    Value b;              // bias
    bool nonlin;

public:
    Neuron(int nin, bool nonlin = true);
    Value operator()(const std::vector<Value> &x); // forward pass
    std::vector<Value *> parameters() override;

    friend std::ostream &operator<<(std::ostream &os, Neuron const &neuron); // << operator
};

#endif // NEURON_H
