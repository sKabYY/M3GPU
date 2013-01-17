#include <cstdio>
#include <cstdlib>
#include <string>

using namespace std;

#include "linear.h"
#include <cuda_runtime.h>
#include <sys/time.h>

bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}

	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	printf("InitCuda succeeded\n");
	return true;
}

void exit_with_help(char *me) {
	printf("Usage: %s test_file model_dir [output_file]\n", me);
	exit(1);
}

long getmilliseconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return ((long)tv.tv_sec*1000 + (long)tv.tv_usec/1000);
}

int main(int argc, char **argv) {
	if (argc != 3 && argc != 4) {
		exit_with_help(argv[0]);
	}
	InitCUDA();
	string dir, infn, outfn;
	infn = argv[1];
	dir = argv[2];
	if (argc == 4) {
		outfn = argv[3];
	} else {
		outfn = "output";
	}
	m3model *model_ = load_model(dir);
	FILE *input, *output;
	input = fopen(infn.c_str(), "r");
	output = fopen(outfn.c_str(), "w");
	long start, t;
	start = getmilliseconds();
	predict_file(model_, input, output);
	t = getmilliseconds() - start;
	printf("PredictTime: %ldms\n", t);
	free_and_destroy_model(model_);
}

