#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <value.hpp>

class Module
{
public:
    virtual void zero_grad();
    virtual std::vector<Value *> parameters() = 0;
};

#endif // MODULE_H
