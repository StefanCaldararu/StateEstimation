#include "particleFilter.h"
#include <random>

particleFilter::particleFilter(){
    dt = 0.1;
    numParticles = 100;
    std::default_random_engine generator;
    std::normal_distribution<double> GPSdistributor(0,0.8);
    std::normal_distribution<double> MAGdistributor(0,0.8);
    //initialize the particles.
    for(int i = 0;i<numParticles;i++){
        particles.push_back(particle{GPSdistributor(generator), GPSdistributor(generator), 0,0,0.01});
        GPSdists.push_back(std::vector<double>());
        MAGdists.push_back(std::vector<double>());
    }
    //generate the distributions distributions
    for(int i = 0;i<100;i++){
        double gval = GPSdistributor(generator);
        double mval = MAGdistributor(generator);
        sampleGPSdist.push_back(gval);
        sampleMAGdist.push_back(mval);
        for(int j = 0;j<numParticles;j++){
            GPSdists[j].push_back(gval);
            MAGdists[j].push_back(mval);
        }
    }


}

state particleFilter::dynamicsModel(state x, input u){
    x.x = x.x+cos(x.theta)*dt*x.v;
    x.y = x.y+sin(x.theta)*dt*x.v;
    x.theta = x.theta+dt*x.v*tan(u.steering)/l;
    double f = tau_0*u.throttle-tau_0*x.v/(omega_0*r_wheel*gamma);
    x.v = x.v+dt*((r_wheel*gamma)/i_wheel)*(f-x.v*c_1)/(r_wheel*gamma-c_0);
    return x;
};

void particleFilter::normalizeWeights(){
    double total = 0;
    for(int i = 0; i<numParticles;i++)
        total = total+particles[i].weight;
    for(int i = 0;i<numParticles;i++)
        particles[i].weight = particles[i].weight/total;
}

void particleFilter::resample(){

}

void particleFilter::update(observation o){

}

state particleFilter::step(input i, observation o){

}

particleFilter::~particleFilter(){

}