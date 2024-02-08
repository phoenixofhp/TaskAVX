#include <iostream>
#include <chrono>
#include <immintrin.h>

uint32_t NonOptFuncMult = 0;
uint32_t OptFuncMult = 0;

static void firNonOptimized(float* x, float* y, float* b, size_t n) {
	std::reverse(b, b + n);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j <= i; j++) {
			y[i] += b[j] * x[j];
			NonOptFuncMult++;
		}
	}
	std::reverse(b, b + n);
}

static void firOptimized(float* x, float* y, float* b, size_t n) {
	std::reverse(b, b + n);
	for (size_t i = 0; i < n; i++) {
		size_t avx256_group_operations = (i + 1) / 8; //avx256 holds 8 floats
		size_t non_optimized_operations = i + 1 - (avx256_group_operations * 8);
		
		for (size_t j = 0; j < avx256_group_operations; j++) {
			__m256 big_group_x, big_group_h, big_group_mult;
			big_group_x = _mm256_load_ps(x + (j * 8));
			big_group_h = _mm256_load_ps(b + (j * 8));

			big_group_mult = _mm256_setzero_ps();
			big_group_mult = _mm256_mul_ps(big_group_x, big_group_h);
			float* big_group_result = (float*)&big_group_mult;
			for (size_t k = 0; k < 8; k++) y[i] += big_group_result[k];
			OptFuncMult++;
		}
		for (size_t j = (avx256_group_operations * 8); j < i + 1; j++) {
			y[i] += x[j] * b[j];
			OptFuncMult++;
		}
	}
	std::reverse(b, b + n);
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

		std::cout << i << '\t' << elapsed_time_non_opt << "\t\t" << elapsed_time_opt\
			<< "\t\t" << NonOptFuncMult << "\t\t" << OptFuncMult << "\t\t" << std::equal(y, y + i, y_comp) << '\n';

		delete[] x;
		delete[] y;
		delete[] y_comp;
		delete[] b;
	}
	//std::cin.get();
}