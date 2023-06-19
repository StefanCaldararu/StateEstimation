#include <iostream>

struct state{
    double x;
    double y;
    double theta;
    double v;
};
struct particle{
    state state;
    double weight;
};

int main(int argc, char ** argv)
{
    particle p = {1.0,2.0,3.0,4.0,5.0};
    std::cout << p.state.theta << std::endl;
    return 1;
}