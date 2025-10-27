/*
 * OpenMP Strategy 5: Advanced Scheduling Optimization
 * 
 * This strategy focuses on optimizing the scheduling of parallel loops
 * using different schedule types (dynamic, guided) and chunk sizes
 * to handle potential load imbalance.
 * 
 * Expected benefits:
 * - Better load balancing for irregular workloads
 * - Adaptive work distribution
 * - Reduced thread idle time
 * 
 * Expected drawbacks:
 * - Higher scheduling overhead
 * - May not benefit regular workloads
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <thread>
#include <complex>
#include <cmath>
#include <omp.h>

typedef std::complex<double> Complex;

#ifndef NFREQ
#define NFREQ (64*1024)
#endif
const long nfreq = NFREQ;

const double dT=0.001;
const long nsamp=2*nfreq;
double dF = 1/(nsamp*dT);

typedef std::vector<Complex> ComplexVector;
typedef std::vector<double> DoubleVector;

DoubleVector read_txt_file(std::string fname) {
    DoubleVector data;
    std::string line;
    std::ifstream file(fname);
    while (std::getline(file, line))
        data.push_back(std::stod(line));
    return data;
}

void fft(std::vector<Complex>& x) {
    const long N = x.size();
    if (N <= 1) return;

    std::vector<Complex> even(N/2), odd(N/2);
    for (long i=0; i<N/2; i++) {
        even[i] = x[2*i];
        odd[i]  = x[2*i+1];
    }

    fft(even);
    fft(odd);

    for (long k = 0; k < N/2; k++) {
        Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k    ] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
}

void ifft(std::vector<Complex>& x) {
    double inv_size = 1.0 / x.size();
    for (auto& xx: x) xx = std::conj(xx);
    fft(x);
    for (auto& xx: x) xx = std::conj(xx) * inv_size;
}

DoubleVector propagator(DoubleVector wave, DoubleVector density, DoubleVector velocity) {
    const long nlayers = density.size();
    DoubleVector imp(nlayers);
    DoubleVector ref(nlayers-1);
    ComplexVector half_filter(nfreq/2+1,1);
    ComplexVector filter(nfreq+1);
    DoubleVector half_wave(nfreq+1,0);
    ComplexVector wave_spectral(nsamp);
    ComplexVector U(nfreq+1,0);
    ComplexVector Upad(nsamp,0);
    DoubleVector seismogram(nsamp);
    long n_wave = wave.size();
    long lc = std::lround(std::floor(nfreq*0.01));
    double mean_wave = 0.;
    
    std::chrono::time_point<std::chrono::high_resolution_clock> tstart1,tstart2,tend1,tend2;
    auto tstart = std::chrono::high_resolution_clock::now();

    // ========================================
    // STRATEGY 5: ADVANCED SCHEDULING OPTIMIZATION
    // Using different scheduling strategies for different types of work
    // ========================================
    
    // Simple computation - use static for minimal overhead
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nlayers; i++)
        imp[i] = density[i] * velocity[i];

    // Simple computation - use static
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nlayers-1; i++)
        ref[i] = (imp[i+1] - imp[i])/(imp[i+1] + imp[i]);

    // Transcendental functions - may benefit from dynamic scheduling
    // as sin() cost might vary
    #pragma omp parallel for schedule(dynamic, 16)
    for (long i=0; i < lc+1; i++)
        half_filter[i]= (sin(M_PI*(2*i-lc)/(2*lc)))/2+0.5;

    // Simple array access - static is optimal
    #pragma omp parallel for schedule(static)
    for (long i=0; i < n_wave/2; i++)
        half_wave[i] = wave[n_wave/2-1+i];

    // Simple array copy - static
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nfreq/2+1; i++)
        filter[i] = half_filter[i];
    
    filter[nfreq/2+1] = 1;
    
    // Array indexing with computation - use guided for adaptive distribution
    #pragma omp parallel for schedule(guided, 32)
    for (long i=nfreq/2+2; i < nfreq+1; i++)
        filter[i] = half_filter[nfreq+1-i];

    // Conditional work - guided scheduling for load balancing
    #pragma omp parallel for reduction(+:mean_wave) schedule(guided, 64)
    for (long i=0; i < 2*nfreq; i++) {
        if (i < nfreq) {
            wave_spectral[i] = half_wave[i];
        } else {
            wave_spectral[i] = half_wave[2*nfreq-i];
        }
        mean_wave += std::real(wave_spectral[i]);
    }

    mean_wave = mean_wave / nsamp;

    // Simple arithmetic - static
    #pragma omp parallel for schedule(static)
    for (long i=0; i < 2*nfreq; i++)
        wave_spectral[i] -= mean_wave;

    // FFT operations (sequential)
    tstart1 = std::chrono::high_resolution_clock::now();
    fft(wave_spectral);
    tend1 = std::chrono::high_resolution_clock::now();

    // Complex computation with varying cost per iteration
    // Different frequencies may have different computational costs
    // Use guided scheduling for adaptive load balancing
    #pragma omp parallel for schedule(guided, 8)
    for (long i=0; i < nfreq+1; i++) {
        Complex omega{0, 2*M_PI*i*dF};
        Complex exp_omega = exp( - dT * omega);
        Complex Y = 0;
        for (long n=nlayers-2; n > -1; n--)
            Y = exp_omega * (ref[n] + Y) / (1.0 + ref[n]*Y);
        U[i] = Y;
    }

    // Simple complex arithmetic - static
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nfreq+1; i++) {
        U[i] *= filter[i];
        Upad[i] = U[i];
    }
    
    // Array access with conjugate - could use dynamic for better balance
    #pragma omp parallel for schedule(dynamic, 32)
    for (long i=nfreq+1; i < nsamp; i++)
        Upad[i] = std::conj(Upad[nsamp - i]);
    
    // Complex multiplication - static is fine
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nsamp; i++)
        Upad[i] *= wave_spectral[i];
    
    // Second FFT (sequential)
    tstart2 = std::chrono::high_resolution_clock::now();
    ifft(Upad);
    tend2 = std::chrono::high_resolution_clock::now();

    // Simple real extraction - static
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nsamp; i++)
        seismogram[i] = std::real(Upad[i]);

    auto tend = std::chrono::high_resolution_clock::now();

    std::cout << "Wave zero-point        : " << std::setw(9) << std::setprecision(5) 
              << mean_wave << "\n";    
    std::cout << "Seismogram first coeff : " << std::setw(9) << std::setprecision(5) 
              << seismogram[0] << ", " << seismogram[1] << ", " << seismogram[2] << ", " << seismogram[3] <<"\n";    
    std::cout << "Elapsed time for FFTs  :" << std::setw(9) << std::setprecision(4)
              << (tend1 - tstart1 + tend2 - tstart2).count()*1e-9 << "\n";
    std::cout << "Elapsed time without FFTs:" << std::setw(9) << std::setprecision(4)
              << (tend - tstart - (tend1 - tstart1 + tend2 - tstart2)).count()*1e-9 << "\n";
    std::cout << "Elapsed time:" << std::setw(9) << std::setprecision(4)
              << (tend - tstart).count()*1e-9 << "\n";
    
    return seismogram;
}

int main(int argc, char* argv[]) {
    std::cout << "=== OpenMP Strategy 5: Advanced Scheduling Optimization ===" << std::endl;
    std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;
    
    DoubleVector wave = read_txt_file("wave_data.txt");
    DoubleVector density = read_txt_file("density_data.txt");
    DoubleVector velocity = read_txt_file("velocity_data.txt");

    DoubleVector seismogram = propagator(wave, density, velocity);

    double checksum = 0;
    for (long i=0; i < nsamp; i++) {
        checksum += std::abs(seismogram[i]);
    }
    std::cout << "Checksum    :" << std::setw(20) << std::setprecision(15)
              << checksum << "\n";

    return 0;
}