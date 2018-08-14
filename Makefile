NVCC := /usr/local/cuda/bin/nvcc --ptxas-options=-v  -arch=compute_60 -code=sm_60 -O3
LIBS := -L/usr/local/cuda/sdk/lib -L/usr/local/cuda/lib
INCS := -I/usr/local/cuda/include -I/usr/include/cuda -I./ -I/usr/local/cuda/sdk/common/inc -I/usr/local/cuda-9.2/targets/x86_64-linux/include
CFLAGS := $(INCS) -c# -D_DEBUG
LDFLAGS := $(LIBS) -lcuda 
#SHA1OBJS := sha1test.o sha1_cpu.o sha1_kernel.o
SHA1OBJS := sha1test.o sha1_kernel.o
PARSHA256OBJS := parsha256test.o parsha256_kernel.o
PARSHA256EMUOBJS := parsha256testemu.o parsha256_kernelemu.o


#all: sha1test parsha256test parsha256testemu
all: sha1test
# SHA-1 benchmark test
sha1test: $(SHA1OBJS)
	$(NVCC) $(LDFLAGS) $(SHA1OBJS) -o sha1test
sha1test.o: sha1test.cu common.h
	$(NVCC) $(CFLAGS) sha1test.cu -o sha1test.o
# sha1_cpu.o: sha1_cpu.cu common.h
# 	$(NVCC) $(CFLAGS) sha1_cpu.cu -o sha1_cpu.o
sha1_kernel.o: sha1_kernel.cu common.h
	$(NVCC) $(CFLAGS) sha1_kernel.cu -o sha1_kernel.o

# # PARSHA-256 benchmark test
# parsha256test: $(PARSHA256OBJS)
# 	$(NVCC) $(LDFLAGS) $(PARSHA256OBJS) -o parsha256test
# parsha256test.o: parsha256test.cu parsha256.h
# 	$(NVCC) $(CFLAGS) parsha256test.cu -o parsha256test.o
# parsha256_kernel.o: parsha256_kernel.cu parsha256.h
# 	$(NVCC) $(CFLAGS) parsha256_kernel.cu -o parsha256_kernel.o

# # PARSHA-256 benchmark test in emulation mode
# parsha256testemu: $(PARSHA256EMUOBJS)
# 	$(NVCC) -deviceemu $(LDFLAGS) $(PARSHA256EMUOBJS) -o parsha256testemu
# parsha256testemu.o: parsha256test.cu parsha256.h
# 	$(NVCC) -deviceemu $(CFLAGS) parsha256test.cu -o parsha256testemu.o
# parsha256_kernelemu.o: parsha256_kernel.cu parsha256.h
# 	$(NVCC) -deviceemu $(CFLAGS) parsha256_kernel.cu -o parsha256_kernelemu.o

clean:
	rm -rf *~
	rm -rf *.o
	rm -rf sha1test
	rm -rf parsha256test
	rm -rf parsha256testemu
