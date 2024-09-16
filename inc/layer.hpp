#ifndef LAYER_H
#define LAYER_H

#include "neuron.hpp"

class Layer : public Module
{
private:
    std::vector<Neuron> neurons;

public:
    Layer(int nin, int nout, bool nonlin = true);
    std::vector<Value> operator()(const std::vector<Value> &x); // forward pass
    std::vector<Value *> parameters() override;
    friend std::ostream &operator<<(std::ostream &os, Layer const &layer); // << operator
};

#endif // LAYER_H
