#include "value.hpp"

std::ostream &operator<<(std::ostream &os, Value const &value)
{
    return os << std::format("Value(data={:.4f}, grad={:.4f})\n", value.data, value.grad);
}

Value::Value()
{
}

// Constructor: initializes the data, children (for autograd), and the operation (op)
Value::Value(double data, std::unordered_set<Value *> children, char op)
{
    this->data = data;         // Set the data (scalar value)
    this->grad = 0;            // Initialize the gradient to 0
    this->_backward = []() {}; // Default empty backward function
    this->_prev = children;    // Track the previous Value objects (for autograd)
    this->_op = op;            // Operation that created this Value
}

// Addition operator
Value Value::operator+(Value other)
{
    Value out(this->data + other.data, {this, const_cast<Value *>(&other)}, '+');

    out._backward = [this, &other, &out]()
    {
        this->grad += out.grad;
        other.grad += out.grad;
    };

    return out;
}

// Subtraction operator
Value Value::operator-(Value other)
{
    Value out(this->data - other.data, {this, const_cast<Value *>(&other)}, '-');

    out._backward = [this, &other, &out]()
    {
        this->grad += out.grad;
        other.grad += out.grad;
    };

    return out;
}

// Unary Minus operator
Value Value::operator-()
{
    return (*this) * -1;
}

// Multiplication operator
Value Value::operator*(Value other)
{
    Value out(this->data * other.data, {this, const_cast<Value *>(&other)}, '*');

    out._backward = [this, &other, &out]()
    {
        this->grad += other.data * out.grad;
        other.grad += this->data * out.grad;
    };

    return out;
}

// Power operator
Value Value::operator^(double other)
{
    Value out(std::pow(this->data, other), {this}, '^');

    out._backward = [this, &other, &out]()
    {
        this->grad += (other * std::pow(this->data, (other - 1))) * out.grad;
    };

    return out;
}

// Division operator
Value Value::operator/(Value other)
{
    return (*this) * (other ^ -1);
}

// ReLU activation function
Value Value::relu()
{
    Value out(this->data < 0 ? 0 : this->data, {this}, 'r'); // 'r' stands for ReLU

    out._backward = [this, &out]()
    {
        this->grad += (out.data > 0 ? 1 : 0) * out.grad;
    };

    return out;
}

// Backward pass
void Value::backward()
{
    // Topological order of all the children in the computation graph
    std::vector<Value *> topo;           // Will hold the sorted graph
    std::unordered_set<Value *> visited; // To keep track of visited nodes

    // Recursive function to perform depth-first search and build topological order
    std::function<void(Value *)> build_topo = [&](Value *v)
    {
        if (visited.find(v) == visited.end())
        {
            visited.insert(v);
            for (Value *child : v->_prev)
            {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    // Build topological ordering starting from 'this' node
    build_topo(this);

    // Set the gradient of the output node (root of computation) to 1
    this->grad = 1.0;

    // Traverse the nodes in reverse topological order to apply chain rule
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
    {
        (*it)->_backward(); // Call each node's backward function
    }
}

// LHS - Value; RHS-double

Value operator+(Value lhs, double rhs)
{
    return lhs + Value(rhs);
}
Value operator-(Value lhs, double rhs)
{
    return lhs - Value(rhs);
}
Value operator*(Value lhs, double rhs)
{
    return lhs * Value(rhs);
}
Value operator/(Value lhs, double rhs)
{
    return lhs / Value(rhs);
}

// LHS - double; RHS-Value

Value operator+(double lhs, Value rhs)
{
    return Value(lhs) + rhs;
}
Value operator-(double lhs, Value rhs)
{
    return Value(lhs) - rhs;
}
Value operator*(double lhs, Value rhs)
{
    return Value(lhs) * rhs;
}
Value operator/(double lhs, Value rhs)
{
    return Value(lhs) / rhs;
}