CXX = nvcc
CFLAGS = -arch sm_13 -g -G -pg
#CFLAGS = -Wall -Wconversion -O3 -fPIC -pg

all: train predict

train: linear.o train.cu
	$(CXX) $(CFLAGS) -o train train.cu linear.o

predict: linear.o predict.cu
	$(CXX) $(CFLAGS) -o predict predict.cu linear.o

linear.o: linear.cu linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cu

tags: linear.h linear.cu train.cu
	ctags linear.h linear.cu train.cu

clean:
	rm -f *~ linear.o train predict
