/*
 * OpenMP Strategy 1: Multiple Parallel Regions
 * 
 * This strategy uses separate #pragma omp parallel for directives
 * for each parallelizable loop. This is the most straightforward
 * approach but creates overhead from multiple thread team creation
 * and destruction.
 * 
 * Expected benefits:
 * - Simple to implement and understand
 * - Each loop gets dedicated parallelization
 * - Good for independent computations
 * 
 * Expected drawbacks:
 * - Thread creation/destruction overhead
 * - Implicit barriers between parallel regions
 * - No opportunity for work sharing between loops
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

// Use standard allocator for simplicity
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
    // STRATEGY 1: MULTIPLE PARALLEL REGIONS
    // ========================================
    
    // Parallel Region 1: Compute seismic impedance
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nlayers; i++)
        imp[i] = density[i] * velocity[i];
    
    // Parallel Region 2: Reflection coefficients
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nlayers-1; i++)
        ref[i] = (imp[i+1] - imp[i])/(imp[i+1] + imp[i]);

    // Parallel Region 3: Spectral window
    #pragma omp parallel for schedule(static)
    for (long i=0; i < lc+1; i++)
        half_filter[i]= (sin(M_PI*(2*i-lc)/(2*lc)))/2+0.5;

    // Parallel Region 4: Filter preparation
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nfreq/2+1; i++)
        filter[i] = half_filter[i];

    filter[nfreq/2+1] = 1;

    #pragma omp parallel for schedule(static)
    for (long i=nfreq/2+2; i < nfreq+1; i++)
        filter[i] = half_filter[nfreq+1-i];

    // Parallel Region 5: Wave processing
    #pragma omp parallel for schedule(static)
    for (long i=0; i < n_wave/2; i++)
        half_wave[i] = wave[n_wave/2-1+i];

    // Parallel Region 6: Wave spectral with reduction
    #pragma omp parallel for reduction(+:mean_wave)
    for (long i=0; i < 2*nfreq; i++) {
        if (i < nfreq) {
            wave_spectral[i] = half_wave[i];
        } else {
            wave_spectral[i] = half_wave[2*nfreq-i];
        }
        mean_wave += std::real(wave_spectral[i]);
    }

    mean_wave = mean_wave / nsamp;

    // Parallel Region 7: Mean subtraction
    #pragma omp parallel for schedule(static)
    for (long i=0; i < 2*nfreq; i++)
        wave_spectral[i] -= mean_wave;

    // FFT (sequential)
    tstart1 = std::chrono::high_resolution_clock::now();
    fft(wave_spectral);
    tend1 = std::chrono::high_resolution_clock::now();

    // Parallel Region 8: Upgoing waves computation
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nfreq+1; i++) {
        Complex omega{0, 2*M_PI*i*dF};
        Complex exp_omega = exp( - dT * omega);
        Complex Y = 0;
        for (long n=nlayers-2; n > -1; n--)
            Y = exp_omega * (ref[n] + Y) / (1.0 + ref[n]*Y);
        U[i] = Y;
    }

    // Parallel Region 9: Apply filter and prepare Upad
    #pragma omp parallel for schedule(static)
    for (long i=0; i < nfreq+1; i++) {
        U[i] *= filter[i];
        Upad[i] = U[i];
    }

    #pragma omp parallel for schedule(static)
    for (long i=nfreq+1; i < nsamp; i++)
        Upad[i] = std::conj(Upad[nsamp - i]);

    #pragma omp parallel for schedule(static)
    for (long i=0; i < nsamp; i++)
        Upad[i] *= wave_spectral[i];
    
    // Second FFT (sequential)
    tstart2 = std::chrono::high_resolution_clock::now();
    ifft(Upad);
    tend2 = std::chrono::high_resolution_clock::now();

    // Parallel Region 10: Final seismogram extraction
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
    std::cout << "=== OpenMP Strategy 1: Multiple Parallel Regions ===" << std::endl;
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