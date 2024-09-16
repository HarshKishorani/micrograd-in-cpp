#include "module.hpp"

void Module::zero_grad()
{
    for(Value* p : this->parameters()){
        p->grad = 0;
    }
}