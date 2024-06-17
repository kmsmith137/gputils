// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

#include <iostream>
#include "../include/gputils.hpp"

using namespace std;
using namespace gputils;


template<typename T>
struct Tracker
{
    int nelts = 0;
    int nmc = 0;
    
    vector<double> sum;
    vector<double> sum2;
    vector<double> sum_imbalance;
    
    Tracker(int nelts_) :
	nelts(nelts_),
	sum(nelts_, 0.0),
	sum2(nelts_, 0.0),
	sum_imbalance((nelts_*(nelts_-1))/2, 0)
    { }

    void add(const vector<T> &v)
    {
	assert(v.size() == nelts);
	nmc++;
	
	for (int i = 0; i < nelts; i++) {
	    sum[i] += v[i];
	    sum2[i] += v[i]*v[i];
	}

	int s = 0;
	for (int i = 0; i < nelts; i++) {
	    for (int j = i+1; j < nelts; j++) {
		if (v[i] < v[j])
		    sum_imbalance[s] -= 1.0;
		else if (v[i] > v[j])
		    sum_imbalance[s] += 1.0;
	    }
	    s++;
	}
    }

    void summarize()
    {
	assert(nmc >= 10);
	
	for (int i = 0; i < nelts; i++) {
	    double mean = sum[i] / nmc;
	    double var = (sum2[i] - nmc*mean*mean) / (nmc-1);
	    cout << "    Component " << i << ": mean = " << mean << ", std = " << sqrt(var) << endl;
	}

	int s = 0;	
	for (int i = 0; i < nelts; i++) {
	    for (int j = i+1; j < nelts; j++) {
		double imbalance = sum_imbalance[s] / nmc;
		cout << "    Components " << i << "," << j << ": imbalance = " << imbalance << endl;
	    }
	    s++;
	}
    }
};


int main(int argc, char **argv)
{
    cout << "Calling random_doubles_with_fixed_sum(4, 100.0)" << endl;
    Tracker<double> dtracker(4);
    
    for (int n = 0; n < 1000*1000; n++) {
	vector<double> v = random_doubles_with_fixed_sum(4, 100.0);
	
	if (n <= 10) {
	    double sum = 0.0;
	    for (uint i = 0; i < v.size(); i++)
		sum += v[i];
	    cout << "    Call " << n << ": " << tuple_str(v," ") << ", sum=" << sum << endl;
	}

	dtracker.add(v);
    }

    dtracker.summarize();

    // -------------------------------------------------------------------------------------------------

    cout << "Calling random_integers_with_bounded_product(4, 1000*1000)" << endl;
    Tracker<ssize_t> itracker(4);
    
    for (int n = 0; n < 1000*1000; n++) {
	vector<ssize_t> v = random_integers_with_bounded_product(4, 1000*1000);
	
	if (n <= 10) {
	    ssize_t product = 1;
	    for (uint i = 0; i < v.size(); i++)
		product *= v[i];
	    cout << "    Call " << n << ": " << tuple_str(v," ") << ", product=" << product << endl;
	}

	itracker.add(v);
    }

    itracker.summarize();
}
