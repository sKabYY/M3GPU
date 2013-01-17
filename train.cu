#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace std;

#include "linear.h"

#include <sys/time.h>
#include <cuda_runtime.h>

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

void print_null(const char *s) {}

void print_usage(char *me) {
  printf(
      "Usage: %s [options] training_set_dir [model_file]\n"
      "options:\n"
      "    -c cost : set the parameter C (default 1)\n"
      "    -e epsilon : set tolerance of termination criterion\n"
      "    -q : quiet mode (no outputs)\n",
      me);
  exit(1);
}

void parse_command_line(int argc, char **argv,
    string *input_dir_name_ptr,
    string *model_dir_name_ptr,
    parameter *param_ptr)
{
  int i;
  void (*print_func)(const char*) = NULL;
  // default values
  param_ptr->C = 1;
  param_ptr->eps = 0.1;

  // parse options
  for(i=1;i<argc;i++) {
    if(argv[i][0] != '-') break;
    if(++i>=argc)
      print_usage(argv[0]);
    switch(argv[i-1][1]) {
      case 'c':
        param_ptr->C = atof(argv[i]);
        break;
      case 'e':
        param_ptr->eps = atof(argv[i]);
        break;
      case 'q':
        print_func = &print_null;
        i--;
        break;
      default:
        fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
        print_usage(argv[0]);
        break;
    }
  }

  set_print_string_function(print_func);

  // determine filenames
  if(i>=argc)
    print_usage(argv[0]);
  *input_dir_name_ptr = argv[i++];

  if(i<argc) {
    *model_dir_name_ptr =argv[i];
  } else {
    *model_dir_name_ptr = "model";
  }
}

long getmilliseconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return ((long)tv.tv_sec*1000 + (long)tv.tv_usec/1000);
}

int main(int argc, char **argv) {
  string input_dir_name;
  string model_dir_name;
  parameter param;
  problem prob;
  parse_command_line(argc, argv, &input_dir_name,
      &model_dir_name, &param);
	long start, t;
	start = getmilliseconds();
	if (!InitCUDA()) {
		exit(1);
	}
	t = getmilliseconds() - start;
	printf("InitTime: %ldms\n", t);
	printf("read_problem...\n");
  read_problem(input_dir_name, &prob);
	printf("ok\n");
  const char *error_msg = check_parameter(&param);
  if(error_msg) {
    fprintf(stderr,"ERROR: %s\n",error_msg);
    exit(1);
  }
	printf("train...\n");
	start = getmilliseconds();
  m3model *model_=train(&prob, &param);
	t = getmilliseconds() - start;
	printf("TrainTime: %ldms\n", t);
	printf("ok\n");
	printf("save_model\n");
	save_model(model_dir_name, model_);
	printf("ok\n");
  free_and_destroy_model(model_);
  free(prob.posModular);
  free(prob.negModular);
  free(prob.data.x);
  free(prob.data.y);
  free(prob.data.x_space);
  return 0;
}
