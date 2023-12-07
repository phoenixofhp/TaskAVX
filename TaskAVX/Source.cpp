#include <iostream>
#include <chrono>
#include <immintrin.h>

//#define GLOBAL_DEBUG
//#define AVX_DEBUG

uint32_t NonOptFuncLoopExec = 0;
uint32_t OptFuncLoopExec = 0;

static void firNonOptimized(float* x, float* y, float* b, size_t n) {
#ifdef GLOBAL_DEBUG
	std::cout << std::endl << "Started non optimized with n=" << n << std::endl;
#endif
	std::reverse(b, b + n);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j <= i; j++) {
			y[i] += b[j] * x[j];
			NonOptFuncLoopExec++;
		}
	}
	std::reverse(b, b + n);
#ifdef GLOBAL_DEBUG
	for (size_t i = 0; i < n; i++) std::cout << "y[" << i << "]=" << y[i] << ' ';
	std::cout << std::endl;
#endif
}

static void firOptimized(float* x, float* y, float* b, size_t n) {
#ifdef GLOBAL_DEBUG
	std::cout << "Started optimized with n=" << n << std::endl;
#endif
	std::reverse(b, b + n);
	for (size_t i = 0; i < n; i++) {
		size_t avx256_group_operations = (i + 1) / 8; //avx256 holds 8 floats
		size_t non_optimized_operations = i + 1 - (avx256_group_operations * 8);
#ifdef AVX_DEBUG
		std::cout << "Optimized AVX operations " << avx256_group_operations << std::endl;
		std::cout << "Non optimized aligned operations " << non_optimized_operations << std::endl;
#endif
		
		for (size_t j = 0; j < avx256_group_operations; j++) {
#ifdef AVX_DEBUG
			std::cout << "Entered AVX" << std::endl;
#endif
			__m256 big_group_x, big_group_h, big_group_mult;
			big_group_x = _mm256_load_ps(x + (j * 8));
			big_group_h = _mm256_load_ps(b + (j * 8));

			big_group_mult = _mm256_setzero_ps();
			big_group_mult = _mm256_mul_ps(big_group_x, big_group_h);
#ifdef AVX_DEBUG
			for (size_t k = 0; k < n; k++) std::cout << "x[" << k << "]=" << x[k] << ' ';
			for (size_t k = 0; k < n; k++) std::cout << "y[" << k << "]=" << y[k] << ' ';
			for (size_t k = 0; k < n; k++) std::cout << "b[" << k << "]=" << b[k] << ' ';
#endif
			float* big_group_result = (float*)&big_group_mult;
			for (size_t k = 0; k < 8; k++) y[i] += big_group_result[k];
#ifdef AVX_DEBUG
			for (size_t k = 0; k < 8; k++) std::cout << big_group_result[k] << ' ';
			for (size_t k = 0; k < n; k++) std::cout << "x[" << k << "]=" << x[k] << ' ';
			for (size_t k = 0; k < n; k++) std::cout << "y[" << k << "]=" << y[k] << ' ';
			for (size_t k = 0; k < n; k++) std::cout << "b[" << k << "]=" << b[k] << ' ';
#endif
			OptFuncLoopExec++;
		}
		for (size_t j = (avx256_group_operations * 8); j < i + 1; j++) {
#ifdef AVX_DEBUG
			for (size_t k = 0; k < n; k++) std::cout << "x[" << k << "]=" << x[k] << ' ';
			for (size_t k = 0; k < n; k++) std::cout << "y[" << k << "]=" << y[k] << ' ';
			for (size_t k = 0; k < n; k++) std::cout << "b[" << k << "]=" << b[k] << ' ';
#endif
			y[i] += x[j] * b[j];
			OptFuncLoopExec++;
		}
	}
	std::reverse(b, b + n);
#ifdef GLOBAL_DEBUG
	for (size_t i = 0; i < n; i++) std::cout << "y[" << i << "]=" << y[i] << ' ';
	std::cout << std::endl;
#endif
}

int main() {

	std::chrono::steady_clock::time_point t1;
	std::chrono::steady_clock::time_point t2;
	double t_diff;
	double elapsed_time_non_opt, elapsed_time_opt;

	t1 = std::chrono::high_resolution_clock::now();
	t2 = std::chrono::high_resolution_clock::now();
	t_diff = std::chrono::duration<double, std::milli>(t1 - t2).count(); //overhead compensation

	std::cout << "Length\tNon optimized\tOptimized\tNon Opt\t\tOpt\t\tIs same\n";

	for (size_t i = 1; i <= 65536; i *= 2) {

		float* x = new float[i];
		float* y = new float[i];
		float* b = new float[i];
		float* y_comp = new float[i];
		/*
		memset(x, 0.3333, i);
		memset(y, 0.0, i);
		memset(b, 0.6666, i);*/
		std::fill(x, x + i, 0.3333);
		std::fill(y, y + i, 0.0);
		std::fill(b, b + i, 0.6666);

		t1 = std::chrono::high_resolution_clock::now();
		firNonOptimized(x, y, b, i);
		t2 = std::chrono::high_resolution_clock::now();
		elapsed_time_non_opt = std::chrono::duration<double, std::milli>(t2 - t1).count() - t_diff;

		std::copy(y, y + i, y_comp);
		std::fill(y, y + i, 0.0);

		t1 = std::chrono::high_resolution_clock::now();
		firOptimized(x, y, b, i);
		t2 = std::chrono::high_resolution_clock::now();
		elapsed_time_opt = std::chrono::duration<double, std::milli>(t2 - t1).count() - t_diff;
#ifdef GLOBAL_DEBUG
		for (size_t j = 0; j < i; j++) std::cout << "y[" << j << "]=" << y[j] << ' ' << std::endl;
		for (size_t j = 0; j < i; j++) std::cout << "y_comp[" << j << "]=" << y_comp[j] << ' '<< std::endl;
#endif
		std::cout << i << '\t' << elapsed_time_non_opt << "\t\t" << elapsed_time_opt\
			<< "\t\t" << NonOptFuncLoopExec << "\t\t" << OptFuncLoopExec << "\t\t" << std::equal(y, y + i, y_comp) << '\n';

		delete[] x;
		delete[] y;
		delete[] y_comp;
		delete[] b;
	}
	//std::cin.get();
}