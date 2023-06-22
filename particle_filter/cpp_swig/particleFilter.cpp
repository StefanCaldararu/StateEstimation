#include "particleFilter.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

particleFilter::particleFilter(){
    dt = 0.1;
    numParticles = 100;
    //TODO: make the random generator actually be random, not seeded!!
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> GPSdistributor(0,0.8);
    std::normal_distribution<double> MAGdistributor(0,0.1);
    //initialize the particles.
    for(int i = 0;i<numParticles;i++){
        particles.push_back(particle{GPSdistributor(generator), GPSdistributor(generator), 0,0,0.01});
        GPSdists.push_back(std::deque<double>());
        MAGdists.push_back(std::deque<double>());
    }
    //generate the distributions distributions
    for(int i = 0;i<100;i++){
        double gval = std::abs(GPSdistributor(generator));
        double mval = std::abs(MAGdistributor(generator));
        sampleGPSdist.push_back(gval);
        sampleMAGdist.push_back(mval);
        for(int j = 0;j<numParticles;j++){
            GPSdists[j].push_back(gval);
            MAGdists[j].push_back(mval);
        }
    }
    //sort the two sample dists.

    sortDist(sampleGPSdist);
    sortDist(sampleMAGdist);
    time = 0;


}

void merge(std::deque<double>& arr, std::deque<double>& left, std::deque<double>& right) {
    size_t leftSize = left.size();
    size_t rightSize = right.size();
    size_t i = 0, j = 0, k = 0;
    while (i < leftSize && j < rightSize) {
        if (left[i] <= right[j]) {
            arr[k] = left[i];
            i++;
        }
         else {
            arr[k] = right[j];
            j++;
        }
        k++;
    }
    while (i < leftSize) {
        arr[k] = left[i];
        i++;
        k++;
    }

    while (j < rightSize) {
        arr[k] = right[j];
        j++;
        k++;
    }
}

// Merge sort implementation
void particleFilter::sortDist(std::deque<double>& arr) {
    size_t size = arr.size();

    if (size < 2) {
        return;
    }

    size_t mid = size / 2;
    std::deque<double> left(arr.begin(), arr.begin() + mid);
    std::deque<double> right(arr.begin() + mid, arr.end());

    sortDist(left);
    sortDist(right);

    merge(arr, left, right);
}


state particleFilter::dynamicsModel(state x, input u){
    x.x = x.x+cos(x.theta)*dt*x.v;
    x.y = x.y+sin(x.theta)*dt*x.v;
    x.theta = x.theta+dt*x.v*tan(u.steering)/l;
    double f = tau_0*u.throttle-tau_0*x.v/(omega_0*r_wheel*gamma);
    x.v = x.v+dt*((r_wheel*gamma)/i_wheel)*(f-(x.v*c_1)/(r_wheel*gamma)-c_0);
    return x;
};

void particleFilter::normalizeWeights(){
    double total = 0;
    for(int i = 0; i<numParticles;i++)
        total = total+particles[i].weight;
    for(int i = 0;i<numParticles;i++)
        particles[i].weight = particles[i].weight/total;
}

void particleFilter::assign_weight(int i){
    std::deque<double> sortedGPSdist = GPSdists[i];
    std::deque<double> sortedMAGdist = MAGdists[i];
    sortDist(sortedGPSdist);
    sortDist(sortedMAGdist);
    double w1 = 0;
    double w2 = 0;
    for(int j = 0;j<100;j++){
        w1 = w1+std::abs(sortedGPSdist[j]-sampleGPSdist[j]);
        w2 = w2+std::abs(sortedMAGdist[j]-sampleMAGdist[j]);
    }
    particles[i].weight = 0;
    if(w1 !=0)
        particles[i].weight += 0.9/w1;
    else
        particles[i].weight+=0.9;
    if(w2!=0)
        particles[i].weight +=0.1/w2;
    else
        particles[i].weight+=0.1;
}

void particleFilter::resample(){
    double cumweight = 0;
    std::vector<double> cummulative_weights;
    for(int i = 0;i<numParticles;i++){
        cumweight = cumweight+particles[i].weight;
        cummulative_weights.push_back(cumweight);
    }
    
    //now have the cummulative weights.
    //start sampling points evenly between 0 and 1.
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<double> sampled_points;
    for(int i =0;i<numParticles;i++){
        double ran = dis(generator);

        bool found = false;
        int it = 0;
        while(!found){
            if(ran<cummulative_weights[it]){
                found = true;
                sampled_points.push_back(it);
            }
            else
                it++;
        }
    }
    //now propogate appropriate particles
    std::vector<int> already_seen;
    std::vector<particle> new_particles;
    std::vector<std::deque<double> > new_Mdists;
    std::vector<std::deque<double> > new_Gdists;

    //for randomizing particle locations
    std::normal_distribution<double> distributor(0,0.1);

    for(int i = 0;i<numParticles;i++){
        int point = sampled_points[i];
        new_Gdists.push_back(GPSdists[point]);
        new_Mdists.push_back(MAGdists[point]);
        auto it = std::find(already_seen.begin(), already_seen.end(), point);
        if (already_seen.empty() || it == already_seen.end()) {
            //then we have not already seen this point...
            already_seen.push_back(point);
            new_particles.push_back(particles[point]);
        }
        else{
            //already has been seen...
            particle p = particles[point];
            p.s.x = p.s.x+distributor(generator);
            p.s.y = p.s.y+distributor(generator);
            new_particles.push_back(p);
        }
    }
    // if(already_seen.size()<100)
    //     std::cout << "RESAMPLED " << already_seen.size() << " POINTS" << std::endl;
    particles = new_particles;
    GPSdists = new_Gdists;
    MAGdists = new_Mdists;

}

void particleFilter::update(observation o, int i){
    double distance = std::sqrt((particles[i].s.x-o.x)*(particles[i].s.x-o.x)+(particles[i].s.y-o.y)*(particles[i].s.y-o.y));
    GPSdists[i].push_back(distance);
    MAGdists[i].push_back(std::abs(o.theta-particles[i].s.theta));
    GPSdists[i].pop_front();
    MAGdists[i].pop_front();
}

state particleFilter::step(input u, observation o){
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distributor(0,0.02);
    for(int i = 0;i<numParticles;i++){
        input myinput = u;
        myinput.steering = myinput.steering+distributor(generator);
        myinput.throttle = myinput.steering+distributor(generator);
        particles[i].s = dynamicsModel(particles[i].s, u);
        update(o, i);
        assign_weight(i);
    }
    normalizeWeights();
    state mystate = {0,0,0,0};
    for(int i = 0;i<numParticles;i++){
        mystate.x = mystate.x+particles[i].s.x*particles[i].weight;
        mystate.y = mystate.y+particles[i].s.y*particles[i].weight;
        mystate.theta = mystate.theta+particles[i].s.theta*particles[i].weight;
        mystate.v = mystate.v+particles[i].s.v*particles[i].weight;
    }
    time++;
    if(time %5 == 0)
        resample();

    return mystate;

}