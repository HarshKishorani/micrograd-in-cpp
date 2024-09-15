#include <iostream>
#include "value.hpp"

int main()
{
    Value value = Value(2, {});
    Value value2 = Value(3, {});
    Value value3 = value * value2;

    std::cout << value3;
    for (Value *v : value3._prev)
    {
        std::cout << *v;
    }

    value3.backward();

    std::cout << "\n"
              << value3;
    for (Value *v : value3._prev)
    {
        std::cout << *v;
    }

    return 0;
}