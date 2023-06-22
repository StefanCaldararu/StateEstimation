#include "particleFilter.h"
#include <iostream>

int main(int argc, char ** argv)
{
    particleFilter pf;
    observation o = {0,0,0};
    input u = {0.8,0.0};
    std::vector<state> predictedStates;
    state predictedState;
    for(int i = 0;i<1000;i++){
        predictedState = pf.step(u,o);
        predictedStates.push_back(predictedState);
        //std::cout << predictedState.x << std::endl;
        o.x = predictedState.x;
        o.y = predictedState.y;
        o.theta = predictedState.theta;
    }
    return 0;
}