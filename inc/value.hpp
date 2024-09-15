#pragma once

#include <iostream>
#include <unordered_set>
#include <format>
#include <cmath>
#include <functional>
#include <vector>

class Value
{
public:
    double data;                       // Data in Value (scalar)
    double grad;                       // Gradient of Value
    std::function<void()> _backward;   // Lambda for the backward function
    std::unordered_set<Value *> _prev; // Set of previous Value pointers (for autograd graph construction)
    char _op;                          // Operation that created this value ('+', '*', etc.)

    // Constructor: initializes data, _prev (children), and _op
    // we can use 'explicit' on constructor to avoid implicit type conversion.
    Value(double data, std::unordered_set<Value *> children = {}, char op = ' ');

    // Overload operators for scalar operations
    Value operator+(Value other); // addition
    Value operator-(Value other); // subtraction
    Value operator-();            // unary minus
    Value operator*(Value other); // multiplication
    Value operator/(Value other); // division
    Value relu();

    Value operator^(double other);
    Value operator^(Value other);

    friend Value operator+(Value lhs, double rhs);
    friend Value operator*(Value lhs, double rhs);
    friend Value operator-(Value lhs, double rhs);
    friend Value operator/(Value lhs, double rhs);

    friend Value operator+(double lhs, Value rhs);
    friend Value operator*(double lhs, Value rhs);
    friend Value operator-(double lhs, Value rhs);
    friend Value operator/(double lhs, Value rhs);

    friend std::ostream &operator<<(std::ostream &os, Value const &value); // << operator

    // Backward function
    void backward();
};