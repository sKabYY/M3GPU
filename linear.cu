#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cmath>
#include <cerrno>

#include <string>
#include <sstream>
#include <fstream>

using namespace std;

#include <sys/stat.h>

#include "linear.h"

#define INF HUGE_VAL
#define MAX_ITER 1000

static void print_string_stdout(const char *s) {
  fputs(s, stdout);
  fflush(stdout);
}

static void (*linear_print_string)(const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt, ...) {
  char buf[BUFSIZ];
  va_list ap;
  va_start(ap, fmt);
  vsprintf(buf, fmt, ap);
  va_end(ap);
  (*linear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

static char* readline(FILE *input) {
  static int max_line_len = 1024;
  int len;
  char *line = Malloc(char,max_line_len);

  if (fgets(line, max_line_len, input) == NULL)
    return NULL;

  while (strrchr(line, '\n') == NULL) {
    max_line_len *= 2;
    line = (char *) realloc(line, max_line_len);
    len = (int) strlen(line);
    if (fgets(line + len, max_line_len - len, input) == NULL)
      break;
  }
  return line;
}

static void exit_input_error(int line_num) {
  fprintf(stderr, "Wrong input format at line %d\n", line_num);
  exit(1);
}


#undef GETI
#define GETI(i) (y[i]+1)
#define SWAP(X, Y) {X+=Y;Y=X-Y;X-=Y;}
__device__ static int randnext(int next) {
	next = next * 1103515245 + 12345;
	return (unsigned int)(next/65536) % 32768;
}

typedef struct cudaproblem {
	double C, eps;
	int numOfPosModulars, numOfNegModulars;
} cudaproblem;

typedef struct infomation {
	int iter;
} infomation;

__global__ static void solve_l2r_l2_svc(
	/*=TEMP=*/
	double *alpha_pool, int *index_pool,
	/*=INPUT=*/
	cudaproblem *cudaprob, 
	double *QD, int *index_i, 
	schar *y, int *x, feature_node *x_space, 
	modular *pmodulars, modular *nmodulars, submodel *subs,
	/*=OUTPUT=*/
	double *wpool, 
	/*=INFOMATION=*/
	infomation *infoma
) {
	int negidx = blockIdx.x;
	int posidx = threadIdx.x;
	modular *pos_modular = &(pmodulars[posidx]);
	modular *neg_modular = &(nmodulars[negidx]);
	int modidx = negidx * cudaprob->numOfPosModulars + posidx;
	int lp = pos_modular->l;
	int ln = neg_modular->l;
	int l = lp + ln;
	double C = cudaprob->C;
	double eps = cudaprob->eps;

	double *alpha = &(alpha_pool[index_i[modidx]]);
	int *index = &(index_pool[index_i[modidx]]);
	double *w = &(wpool[subs[modidx].offset]);

	int iter = 0;
	double d, G;
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	const double diag = 0.5/C;

	for(int ii=0; ii<l; ii++)
	{
		modular *mptr;
		int i = ii;
		if (i < lp) {
			mptr = pos_modular;
			i = ii + mptr->first;
		} else {
			mptr = neg_modular;
			i = ii - lp + mptr->first;
		}
		feature_node *xi = x_space+x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			w[xi->index-1] += y[i]*alpha[ii]*val;
			xi++;
		}
		index[ii] = ii;
	}

	int rseed = 1;
	while (iter < MAX_ITER)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

#if 1
		for (int i=0; i<active_size; i++)
		{
			rseed = randnext(rseed);
			int j = i+rseed%(active_size-i);
			SWAP(index[i], index[j]);
		}

		for (int s=0; s<active_size; s++)
		{
			int ii = index[s];
			modular *mptr;
			int i;
			if (ii < lp) {
				mptr = pos_modular;
				i = ii + mptr->first;
			} else {
				mptr = neg_modular;
				i = ii - lp + mptr->first;
			}
			G = 0;
			schar yi = y[i];

			feature_node *xi = x_space+x[i];
			while(xi->index!= -1)
			{
				G += w[xi->index-1]*(xi->value);
				xi++;
			}
			G = G*yi-1;

			G += alpha[ii]*diag;

			PG = 0;
			if (alpha[ii] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					SWAP(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[ii];
				double alpha_new = max(alpha[ii] - G/QD[i], 0.0);
				alpha[ii] = alpha_new;
				d = (alpha_new - alpha_old)*yi;
				xi = x_space+x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}
#endif

		iter++;
		//if(iter % 10 == 0)
		//	info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				//info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}
	infoma[modidx].iter = iter;
/*
	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= MAX_ITER)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");
*/

	// calculate objective value

	/*
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
	*/
}

#define CudaMalloc(p, type, n) cudaMalloc((void**)&(p), sizeof(type)*(n))
#define CudaMemcpyToDevice(dst, src, type, n) \
	cudaMemcpy(dst, src, sizeof(type)*(n), cudaMemcpyHostToDevice)
#define CudaMallocAndCpy(dst, src, type, n) do {\
		CudaMalloc(dst, type, n); \
		CudaMemcpyToDevice(dst, src, type, n); \
	} while(0)
#define CudaMemcpyToHost(dst, src, type, n) \
	cudaMemcpy(dst, src, sizeof(type)*(n), cudaMemcpyDeviceToHost)
static void train_cuda(const problem *prob, m3model *model_) {
	int *index_i;
	double *QD;
  int np = prob->numOfPosModulars;
  int nn = prob->numOfNegModulars;
	index_i = Malloc(int, np*nn);
	int p = 0;
  for (int i = 0; i < nn; ++i) { // i is neg
    for (int j = 0; j < np; ++j) { // j is pos
			int size = prob->posModular[j].l + prob->negModular[i].l;
			index_i[i*np+j] = p;
			p += size;
    }
  }
	QD = Malloc(double, prob->data.l);
	int n = np+nn;
	int k = 0;
	feature_node *x_space = prob->data.x_space;
	int *indx = prob->data.x;
	for (int i = 0; i < n; ++i) {
		modular *modular_ptr;
		int l;
    if (i < np) {
      modular_ptr = &(prob->posModular[i]);
    } else {
      modular_ptr = &(prob->negModular[i-np]);
    }
		int first = modular_ptr->first;
		l = modular_ptr->l;
		for (int j = 0; j < l; ++j) {
			QD[k] = 0.5/model_->param.C;
			feature_node *xi = x_space+indx[first+j];
			while (xi->index != -1) {
				double val = xi->value;
				QD[k] += val*val;
				++xi;
			}
			++k;
		}
	}

	info("Copying parameters...\n");
	cudaproblem cudaprob;
	cudaprob.C = model_->param.C;
	cudaprob.eps = model_->param.eps;
	cudaprob.numOfPosModulars = np;
	cudaprob.numOfNegModulars = nn;
	cudaproblem *cuda_cudaprob;
	CudaMallocAndCpy(cuda_cudaprob, &cudaprob, cudaproblem, 1);
	const data_pool *data = &(prob->data);
	int l = data->l;
	double *cuda_QD, *cuda_alpha;
	int *cuda_index, *cuda_index_i;
	CudaMalloc(cuda_alpha, double, p);
	cudaMemset(cuda_alpha, 0, sizeof(double)*p);
	CudaMalloc(cuda_index, int, p);
	CudaMallocAndCpy(cuda_QD, QD, double, l);
	CudaMallocAndCpy(cuda_index_i, index_i, int, np*nn);
	schar *cuda_y;
	int *cuda_x;
	feature_node *cuda_x_space;
	modular *cuda_pmodulars, *cuda_nmodulars;
	submodel *cuda_subs;
	double *cuda_w; // This is output.
	CudaMalloc(cuda_w, double, model_->wsize);
	cudaMemset(cuda_w, 0, sizeof(double)*(model_->wsize));
	CudaMallocAndCpy(cuda_x_space, data->x_space, feature_node, data->elements);
	CudaMallocAndCpy(cuda_y, data->y, schar, l);
	CudaMallocAndCpy(cuda_x, data->x, int, l);
	CudaMallocAndCpy(cuda_pmodulars, prob->posModular, modular, np);
	CudaMallocAndCpy(cuda_nmodulars, prob->negModular, modular, nn);
	CudaMallocAndCpy(cuda_subs, model_->subs, submodel, np*nn);
	infomation *cuda_infoma;
	CudaMalloc(cuda_infoma, infomation, np*nn);
	size_t remainingMemory, totalMemory;
	cudaMemGetInfo(&remainingMemory, &totalMemory);
	info("MemInfo: %u/%u\n", remainingMemory, totalMemory);
	info("Enter kernel function<<<%d,%d,0>>>...\n", nn, np);
	solve_l2r_l2_svc<<<nn,np,0>>>(
			/*=TEMP=*/
			cuda_alpha, cuda_index,
			/*=INPUT=*/
			cuda_cudaprob, 
			cuda_QD, cuda_index_i, 
			cuda_y, cuda_x, cuda_x_space, 
			cuda_pmodulars, cuda_nmodulars, cuda_subs,
			/*=OUTPUT=*/
			cuda_w,
			/*=INFOMATION=*/
			cuda_infoma
	);
	CudaMemcpyToHost(model_->w, cuda_w, double, model_->wsize);
	infomation *infoma = Malloc(infomation, np*nn);
	CudaMemcpyToHost(infoma, cuda_infoma, infomation, np*nn);
	for (int i = 0; i < nn; ++i) {
		for (int j = 0; j < np; ++j) {
			int k = i*np+j;
			info("%dvs%d: iter=%d\n", j, i, infoma[k].iter);
		}
	}
	free(infoma);
	cudaFree(cuda_alpha);
	cudaFree(cuda_index);
	cudaFree(cuda_QD);
	cudaFree(cuda_index_i);
	cudaFree(cuda_w);
	cudaFree(cuda_y);
	cudaFree(cuda_x);
	cudaFree(cuda_x_space);
	cudaFree(cuda_pmodulars);
	cudaFree(cuda_nmodulars);
	cudaFree(cuda_subs);
	cudaFree(cuda_infoma);
}

void read_problem(string dir, problem *prob_ptr) {
  ostringstream infofn;
  infofn << dir << "/info";
  ifstream info_file(infofn.str().c_str());
  int numOfPosModulars, numOfNegModulars;
  string item;
  info_file >> item >> numOfPosModulars;
  info_file >> item >> numOfNegModulars;
  info_file.close();
  prob_ptr->numOfPosModulars = numOfPosModulars;
  prob_ptr->numOfNegModulars = numOfNegModulars;
  prob_ptr->posModular = Malloc(modular, numOfPosModulars);
  prob_ptr->negModular = Malloc(modular, numOfNegModulars);
  int totalModulars = numOfPosModulars + numOfNegModulars;
  int elements = 0;
  int l = 0;
  for (int i = 0; i < totalModulars; ++i) {
    int midx;
    ostringstream fn;
    modular *modular_ptr;
    if (i < numOfPosModulars) {
      midx = i;
      modular_ptr = &(prob_ptr->posModular[midx]);
      fn << dir << "/pos." << midx;
    } else {
      midx = i - numOfPosModulars;
      modular_ptr = &(prob_ptr->negModular[midx]);
      fn << dir << "/neg." << midx;
    }
    FILE *fp = fopen(fn.str().c_str(), "r");
    if (fp == NULL) {
      fprintf(stderr, "can't open input file %s\n", fn.str().c_str());
      exit(1);
    }
    char *line;
    int line_counter = 0;
    while ((line = readline(fp)) != NULL) {
      char *p = strtok(line, " \t"); // label
      // features
      while (1) {
        p = strtok(NULL, " \t");
        if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
          break;
        elements++;
      }
      elements++; // for terminal(index=-1) term
      free(line);
      ++line_counter;
    }
    fclose(fp);
    modular_ptr->l = line_counter;
    modular_ptr->first = l;
    l += line_counter;
  }

	prob_ptr->data.l = l;
  prob_ptr->data.y = Malloc(schar,l);
  prob_ptr->data.x = Malloc(int,l);
  feature_node *x_space = Malloc(struct feature_node,elements);
  prob_ptr->data.x_space = x_space;
  prob_ptr->data.elements = elements;
  data_pool *data = &(prob_ptr->data);

  int k = 0, j = 0;
  for (int i = 0; i < totalModulars; ++i) {
    int label, midx;
    ostringstream fn;
    modular *modular_ptr;
    if (i < numOfPosModulars) {
      label = 1;
      midx = i;
      modular_ptr = &(prob_ptr->posModular[midx]);
      fn << dir << "/pos." << midx;
    } else {
      label = -1;
      midx = i - numOfPosModulars;
      modular_ptr = &(prob_ptr->negModular[midx]);
      fn << dir << "/neg." << midx;
    }
    FILE *fp = fopen(fn.str().c_str(), "r");
    if (fp == NULL) {
      fprintf(stderr, "can't open input file %s\n", fn.str().c_str());
      exit(1);
    }
    int max_index = 0;
    char *endptr;
    for (int ii = 0; ii < modular_ptr->l; ++ii, ++k) {
      int inst_max_index = 0; // strtol gives 0 if wrong format
      char *line = readline(fp);
      data->x[k] = j;
      char *label_str = strtok(line, " \t\n");
      if (label_str == NULL) // empty line
        exit_input_error(ii + 1);
      int label_tmp = strtod(label_str, &endptr);
      if (endptr == label_str || *endptr != '\0' || label_tmp != label)
        exit_input_error(ii+1);
      data->y[k] = label;
      while (1) {
        char *idx = strtok(NULL, ":");
        char *val = strtok(NULL, " \t");

        if (val == NULL)
          break;

        errno = 0;
        x_space[j].index = (int) strtol(idx, &endptr, 10);
        if (endptr == idx || errno != 0 || *endptr != '\0'
            || x_space[j].index <= inst_max_index)
          exit_input_error(i + 1);
        else
          inst_max_index = x_space[j].index;

        errno = 0;
        x_space[j].value = strtod(val, &endptr);
        if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
          exit_input_error(i + 1);

        ++j;
      }

      if (inst_max_index > max_index)
        max_index = inst_max_index;

      x_space[j++].index = -1;
      free(line);
    }

    modular_ptr->n = max_index;
    info("modular: %d<%d>, offset=%d, length=%d, index=%d\n",
        label, midx, modular_ptr->first, modular_ptr->l, modular_ptr->n);
  }
}

m3model* train(const problem *prob, const parameter *param) {
  m3model *model_ = Malloc(m3model,1);
  int np = prob->numOfPosModulars;
  int nn = prob->numOfNegModulars;
  model_->numOfPosModulars = np;
  model_->numOfNegModulars = nn;
  model_->subs = Malloc(submodel, np*nn);
  int wsize = 0;
  for (int i = 0; i < nn; ++i) { // i is neg
    for (int j = 0; j < np; ++j) { // j is pos
      int n = max(prob->posModular[j].n, prob->negModular[i].n);
      model_->subs[i*np+j].offset = wsize;
      model_->subs[i*np+j].nr_feature = n;
      wsize += n;
    }
  }
	model_->wsize = wsize;
  model_->w = Malloc(double, wsize);
  model_->param.C = param->C;
  model_->param.eps = param->eps;
  train_cuda(prob, model_);
  return model_;
}

void free_and_destroy_model(m3model *model_) {
  if (model_) {
    if (model_->w)
      free(model_->w);
    if (model_->subs)
      free(model_->subs);
    free(model_);
  }
}

const char *check_parameter(const parameter *param) {
  if (param->eps <= 0)
    return "eps <= 0";

  if (param->C <= 0)
    return "C <= 0";

  return NULL;
}

void set_print_string_function(void (*print_func)(const char*))
{
  if (print_func == NULL)
    linear_print_string = &print_string_stdout;
  else
    linear_print_string = print_func;
}

static void do_save_model(int modidx, const char *fn, 
		const m3model *model_) {
	submodel *smod = &(model_->subs[modidx]);
	int nr_feature = smod->nr_feature;
	double *w = &(model_->w[smod->offset]);
	FILE *fp = fopen(fn, "w");
  if (fp == NULL) {
		fprintf(stderr, "Error: cannot open file %s\n", fn);
		exit(1);
	}
  char *old_locale = strdup(setlocale(LC_ALL, NULL));
  setlocale(LC_ALL, "C");
  fprintf(fp, "nr_feature %d\n", nr_feature);
  fprintf(fp, "w\n");
  for (int i = 0; i < nr_feature; i++) {
    fprintf(fp, "%.16g ", w[i]);
    fprintf(fp, "\n");
  }
  setlocale(LC_ALL, old_locale);
  free(old_locale);
  if (ferror(fp) != 0 || fclose(fp) != 0) {
		fprintf(stderr, "Error: cannot write to file %s\n", fn);
		exit(1);
	}
}

void save_model(string dir, const m3model *model_) {
	if (mkdir(dir.c_str(), 0755) != 0) {
		fprintf(stderr, "Error: cannot mkdir %s\n", dir.c_str());
		exit(1);
	}
	int np = model_->numOfPosModulars;
	int nn = model_->numOfNegModulars;
	ostringstream infofnss;
	infofnss << dir << "/info";
	ofstream infofp(infofnss.str().c_str());
	infofp << "pos: " << np << endl;
	infofp << "neg: " << nn << endl;
	infofp.close();
	for (int i = 0; i < nn; ++i) {
		for (int j = 0; j < np; ++j) {
			ostringstream fnss;
			fnss << dir << "/" << j << "vs" << i;
			do_save_model(i*np+j, fnss.str().c_str(), model_);
		}
	}
}

__global__ static void do_predict(
		schar *res,
		int *np_ptr, int *nn_ptr, submodel *subs, double *wpool,
		feature_node *x
) {
	int np = *np_ptr;
	int negidx = blockIdx.x;
	int posidx = threadIdx.x;
	int modidx = negidx * np + posidx;
	submodel *sm = &(subs[modidx]);
	double *w = &(wpool[sm->offset]);
	int wsize = sm->nr_feature;
	double val = 0;
	while (x->index != -1) {
		if (x->index <= wsize)
			val += w[x->index-1]*x->value;
		++x;
	}
	res[modidx] = (val>0)?1:-1;
	__syncthreads();
	/*
	if (posidx == 0) {
		//TODO
	}*/
}

int predict(
		schar *cuda_res, int *cuda_np, int *cuda_nn, 
		submodel *cuda_subs, double *cuda_w,
		int np, int nn, 
		const feature_node *x, int nr_node) {
	feature_node *cuda_x;
	CudaMallocAndCpy(cuda_x, x, feature_node, nr_node);
	do_predict<<<nn, np, 0>>>(cuda_res, 
			cuda_np, cuda_nn, cuda_subs, cuda_w,
			cuda_x);
	cudaFree(cuda_x);
	schar *res = Malloc(schar, np*nn);
	CudaMemcpyToHost(res, cuda_res, schar, np*nn);
	int val = 1;
	for (int i = 0; i < nn; ++i) {
		bool has_one;
		for (int j = 0; j < np; ++j) {
			if (res[i*np+j] == 1) {
				has_one = true;
				break;
			}
		}
		if (!has_one) {
			val = -1;
			break;
		}
	}
	free(res);
	return val;
}

void predict_file(m3model *model_, FILE *input, FILE *output) {
	/*=InitCUDAMem=*/
	int np = model_->numOfPosModulars;
  int nn = model_->numOfNegModulars;
	schar *cuda_res;
	CudaMalloc(cuda_res, schar, np*nn);
	submodel *cuda_subs;
	double *cuda_w;
	int *cuda_np, *cuda_nn;
	CudaMallocAndCpy(cuda_w, model_->w, double, model_->wsize);
	CudaMallocAndCpy(cuda_subs, model_->subs, submodel, np*nn);
	CudaMallocAndCpy(cuda_np, &np, int, 1);
	CudaMallocAndCpy(cuda_nn, &nn, int, 1);
	/*==*/

	int correct = 0;
	int total = 0;

	char *line;
	int max_nr_attr = 1024;
	feature_node *x = Malloc(feature_node, max_nr_attr);
	while ((line=readline(input)) != NULL) {
		int label;
		label = strtod(strtok(line, " \t\n"), NULL);
		int i = 0;
		while (1) {
			if (i >= max_nr_attr) {
				max_nr_attr *= 2;
				x = (feature_node*)realloc(x, max_nr_attr*sizeof(feature_node));
			}
			char *idx = strtok(NULL, ":");
			char *val = strtok(NULL, " \t");
			if (val == NULL)
				break;
			x[i].index = (int)strtol(idx, NULL, 10);
			x[i].value = strtod(val, NULL);
			++i;
		}
		x[i++].index = -1;
		int target_label;
		
		target_label = predict(
				cuda_res, cuda_np, cuda_nn, 
				cuda_subs, cuda_w,
				np, nn, 
				x, i);
		fprintf(output, "%d\n", target_label);
		if (target_label == label) {
			++correct;
		}
		++total;
		free(line);
	}
	free(x);
	info("Accuracy = %g%% (%d/%d)\n", (double)correct/total*100, 
			correct, total);

	/*=CUDAFree=*/
	cudaFree(cuda_res);
	cudaFree(cuda_np);
	cudaFree(cuda_nn);
	cudaFree(cuda_subs);
	cudaFree(cuda_w);
	/*==*/
}

m3model *load_model(string dir) {
	ostringstream infofnss;
	infofnss << dir << "/info";
	ifstream infofp(infofnss.str().c_str());
	string strtmp;
	int np, nn;
	infofp >> strtmp >> np;
	infofp >> strtmp >> nn;
	infofp.close();
	m3model *model_ = Malloc(m3model, 1);
	model_->numOfPosModulars = np;
	model_->numOfNegModulars = nn;
	model_->subs = Malloc(submodel, np*nn);
	int wsize = 0;
	for (int i = 0; i < nn; ++i) {
		for (int j = 0; j < np; ++j) {
			ostringstream oss;
			oss << dir << "/" << j << "vs" << i;
			ifstream fp(oss.str().c_str());
			int nr_feature;
			fp >> strtmp >> nr_feature;
			fp.close();
			model_->subs[i*np+j].offset = wsize;
			model_->subs[i*np+j].nr_feature = nr_feature;
			wsize += nr_feature;
		}
	}
	model_->w = Malloc(double, wsize);
	model_->wsize = wsize;
	wsize = 0;
	for (int i = 0; i < nn; ++i) {
		for (int j = 0; j < np; ++j) {
			ostringstream oss;
			oss << dir << "/" << j << "vs" << i;
			ifstream fp(oss.str().c_str());
			int nr_feature;
			fp >> strtmp >> nr_feature;
			double *ww = &(model_->w[wsize]);
			fp >> strtmp;
			for (int k = 0; k < nr_feature; ++k) {
				double r;
				fp >> r;
				ww[k] = r;
			}
			fp.close();
			wsize += nr_feature;
		}
	}
	return model_;
}

