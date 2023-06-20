/* File: particleFilter.i */
%module particleFilter

%{
#define SWIG_FILE_WITH_INIT
#include "particleFilter.h"
%}

struct state{
    double x;
    double y;
    double theta;
    double v;
};
struct input{
    double throttle;
    double steering;
};
struct observation{
    double x;
    double y;
    double theta;
};
struct particle{
    state s;
    double weight;
};

class particleFilter
{
    private:
    //timestep
    double dt = 0.1;
    int time;
    //the number of particles
    int numParticles;
    //the dynamics model for the vehicle (4DOF)
    state dynamicsModel(state x, input u);

    //sample distributions of data for GPS and MAG
    std::vector<double> sampleGPSdist;
    std::vector<double> sampleMAGdist;

    //distributions for each of the particles.
    std::vector<std::deque<double> > GPSdists;
    std::vector<std::deque<double> > MAGdists;

    //the states for all of the particels
    std::vector <particle> particles;

    //The weigths for all of the particles
    std::vector <double> weights;

    //normalize the weights of all the particles
    void normalizeWeights();

    //resample the distribution of particles and repopulate likely ones
    void resample();

    //assign weight to particle i
    void assign_weight(observation o, int i);

    //update the distribution for each particle (TODO:or just an individual partclie?)
    void update(observation o, int i);

    //vehicle parameters
    double l = 0.5;
    double tau_0 = 0.09;
    double omega_0 = 161.185;
    double r_wheel = 0.08451952624;
    double gamma = 0.33333333;
    double c_0 = 0.039;
    double c_1 = 1e-4;
    double i_wheel = 1e-3;

    public:
        //constructor
        particleFilter();

        //step the entire filter forwards one timestep
        state step(input u, observation o);

};