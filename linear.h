#ifndef _LINEAR_H
#define _LINEAR_H

#include <string>
using namespace std;

typedef signed char schar;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef struct feature_node
{
  int index;
  double value;
} feature_node;

typedef struct modular
{
  int l, n, first;
} modular;

typedef struct data_pool
{
  int l;
  schar *y;
  int *x;
  int elements;
  feature_node *x_space;
} data_pool;

typedef struct problem
{
  int numOfPosModulars, numOfNegModulars;
  modular *posModular, *negModular;
  data_pool data;
} problem;

typedef struct parameter
{
  /* these are for training only */
  double eps;         /* stopping criteria */
  double C;
} parameter;

typedef struct submodel
{
  int offset;
  int nr_feature;
} submodel;

typedef struct m3model
{
  parameter param;
  int numOfPosModulars, numOfNegModulars;
  submodel *subs;
	int wsize;
  double *w;
} m3model;

void read_problem(string dir, problem *prob_ptr);
void set_print_string_function(void (*print_func) (const char*));
const char *check_parameter(const parameter *param);
void free_and_destroy_model(m3model *model_);
void save_model(string dir, const m3model *model_);
m3model *load_model(string dir);

m3model* train(const problem *prob, const parameter *param);
void predict_file(m3model *model_, FILE *input, FILE *output);

#endif /* _LINEAR_H */
