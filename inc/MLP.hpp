#ifndef MLP_H
#define MLP_H

#include "layer.hpp"

class MLP : public Module
{
private:
    std::vector<Layer> layers;

public:
    MLP(int nin, const std::vector<int> &nouts);
    std::vector<Value> operator()(std::vector<Value> &x); // forward pass
    std::vector<Value *> parameters() override;
    friend std::ostream &operator<<(std::ostream &os, MLP const &mlp); // << operator
};

#endif // MLP_H
