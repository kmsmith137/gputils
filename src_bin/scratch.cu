// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

#include <iostream>
#include "../include/gputils.hpp"

using namespace std;
using namespace gputils;


static void show_permutations(int mx, int my, int mz, int nthreads_per_block)
{
    dim3 nblocks, nthreads;

    // Show all 6 permutations
    assign_kernel_dims(nblocks, nthreads, 32*mx, my, mz, nthreads_per_block, true);
    assign_kernel_dims(nblocks, nthreads, 32*mx, mz, my, nthreads_per_block, true);
    assign_kernel_dims(nblocks, nthreads, 32*my, mx, mz, nthreads_per_block, true);
    assign_kernel_dims(nblocks, nthreads, 32*my, mz, mx, nthreads_per_block, true);
    assign_kernel_dims(nblocks, nthreads, 32*mz, mx, my, nthreads_per_block, true);
    assign_kernel_dims(nblocks, nthreads, 32*mz, my, mx, nthreads_per_block, true);
}


int main(int argc, char **argv)
{
    cout << "Should find nwarps=(1,4,8) and overhead=0.52381\n";
    show_permutations(3, 5, 7, 1024);     // should find nwarps=(2,2,8)

    cout << "\nShould find nwarps=(2,4,4) and overhead=0.174825\n";
    show_permutations(4, 11, 13, 1024);   // should find nwarps=(4,4,2)

    cout << "\nShould find nwarps=(1,2,2) and overhead=0\n";
    show_permutations(2, 6, 15, 128);     // should find nwarps=(2,2,1)

    cout << "\nShould find nwarps=(1,2,2) and overhead=0.2\n";
    show_permutations(2, 3, 5, 128);      // should find nwarps=(2,1,2)

    cout << "\nShould find nwarps=(1,1,4) and overhead=0.142857\n";
    show_permutations(7, 9, 13, 128);     // should find nwarps=(2,1,1)

    cout << "\nShould find nwarps=(1,2,2) and overhead=0.196581\n";
    show_permutations(5, 9, 13, 128);     // should find nwarps=(1,2,2)
    
    return 0;
}
