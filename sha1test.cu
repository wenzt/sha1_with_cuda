/*
 * SHA-1 benchmark program. Calculates execution time of SHA-1 on CPU and GPU.
 * Also includes function sha1_gpu_global() which prepares SHA-1 to be executed
 * on GPU.
 *
 * 2008, Tadas Vilkeliskis <vilkeliskis.t@gmail.com>
 */
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include "common.h"
#include <cuda_profiler_api.h>

#define MAX_THREADS_PER_BLOCK 256
#define CUT_SAFE_CALL(call)\
    cudaError err = call;\
    if (err != cudaSuccess) {\
        fprintf(stderr, "Cut error in file '%s' in line %i.\n", __FILE__, __LINE__);\
        printf("%d\n", err); \
    exit(1);\
}

typedef struct __align__(16) {
	unsigned int state[5];
} sha1_gpu_context;


typedef struct {
	unsigned const char *data;
	unsigned const char *hash;
} testvector;


typedef struct {
	struct timeval  kernel_timer;	/* time spent in kernel */
	struct timeval  malloc_timer;	/* how much time we spend allocating memory */
	struct timeval  memcpy_h2d_timer;	/* how much time we spend copying from host to device */
	struct timeval  memcpy_d2h_timer;	/* how much time we spend copying from device to host */
	struct timeval  free_timer;	/* how much time we spend releasing memory */
	struct timeval  total_timer;	/* how much time we spend total */
} chronometer;

typedef struct {
	double  kernel_timer_res;	
	double  malloc_timer_res;	
	double  memcpy_h2d_timer_res;
	double  memcpy_d2h_timer_res;	
	double  free_timer_res;	
	double  total_timer_res;
} timer_result_t;

/* timers used to check performance */
chronometer chmeter;
timer_result_t time_res;
__constant__ int dblocks;

//extern __global__ void sha1_kernel_global (const unsigned char  *data, sha1_gpu_context  *ctx, unsigned int *extended);
void sha1_cpu_process (sha1_gpu_context *tmp, unsigned int *W);

inline void cutStartTimer(struct timeval *timer)
{
	gettimeofday(timer, NULL);
}

inline void cutStopTimer(struct timeval *timer, double *time_r)
{
	double tmp = (double)timer->tv_sec*1000 + (double)timer->tv_usec / 1000;
	gettimeofday(timer, NULL);
	*time_r = (double)timer->tv_sec*1000 + (double)timer->tv_usec / 1000 - tmp;
}
#define S(x,n) ((x << n) | ((x) >> (32 - n)))
__host__ __device__ void sha1_gpu_process1 (sha1_gpu_context *tmp, unsigned int *W)
{
	unsigned int A, B, C, D, E;
	A = tmp->state[0];
	B = tmp->state[1];
	C = tmp->state[2];
	D = tmp->state[3];
	E = tmp->state[4];

#define P(a,b,c,d,e,x)                                  \
{                                                       \
    e += S(a,5) + F(b,c,d) + K + x; b = S(b,30);        \
}


#define F(x,y,z) (z ^ (x & (y ^ z)))
#define K 0x5A827999
  
	P( A, B, C, D, E, W[0]  );
	P( E, A, B, C, D, W[1]  );
	P( D, E, A, B, C, W[2]  );
	P( C, D, E, A, B, W[3]  );
	P( B, C, D, E, A, W[4]  );
	P( A, B, C, D, E, W[5]  );
	P( E, A, B, C, D, W[6]  );
	P( D, E, A, B, C, W[7]  );
	P( C, D, E, A, B, W[8]  );
	P( B, C, D, E, A, W[9]  );
	P( A, B, C, D, E, W[10] );
	P( E, A, B, C, D, W[11] );
	P( D, E, A, B, C, W[12] );
	P( C, D, E, A, B, W[13] );
	P( B, C, D, E, A, W[14] );
	P( A, B, C, D, E, W[15] );
	P( E, A, B, C, D, W[16] );
	P( D, E, A, B, C, W[17] );
	P( C, D, E, A, B, W[18] );
	P( B, C, D, E, A, W[19] );
  
#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0x6ED9EBA1
  
	P( A, B, C, D, E, W[20] );
	P( E, A, B, C, D, W[21] );
	P( D, E, A, B, C, W[22] );
	P( C, D, E, A, B, W[23] );
	P( B, C, D, E, A, W[24] );
	P( A, B, C, D, E, W[25] );
	P( E, A, B, C, D, W[26] );
	P( D, E, A, B, C, W[27] );
	P( C, D, E, A, B, W[28] );
	P( B, C, D, E, A, W[29] );
	P( A, B, C, D, E, W[30] );
	P( E, A, B, C, D, W[31] );
	P( D, E, A, B, C, W[32] );
	P( C, D, E, A, B, W[33] );
	P( B, C, D, E, A, W[34] );
	P( A, B, C, D, E, W[35] );
	P( E, A, B, C, D, W[36] );
	P( D, E, A, B, C, W[37] );
	P( C, D, E, A, B, W[38] );
	P( B, C, D, E, A, W[39] );
  
#undef K
#undef F

#define F(x,y,z) ((x & y) | (z & (x | y)))
#define K 0x8F1BBCDC

	P( A, B, C, D, E, W[40] );
	P( E, A, B, C, D, W[41] );
	P( D, E, A, B, C, W[42] );
	P( C, D, E, A, B, W[43] );
	P( B, C, D, E, A, W[44] );
	P( A, B, C, D, E, W[45] );
	P( E, A, B, C, D, W[46] );
	P( D, E, A, B, C, W[47] );
	P( C, D, E, A, B, W[48] );
	P( B, C, D, E, A, W[49] );
	P( A, B, C, D, E, W[50] );
	P( E, A, B, C, D, W[51] );
	P( D, E, A, B, C, W[52] );
	P( C, D, E, A, B, W[53] );
	P( B, C, D, E, A, W[54] );
	P( A, B, C, D, E, W[55] );
	P( E, A, B, C, D, W[56] );
	P( D, E, A, B, C, W[57] );
	P( C, D, E, A, B, W[58] );
	P( B, C, D, E, A, W[59] );
  
#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0xCA62C1D6
  
	P( A, B, C, D, E, W[60] );
	P( E, A, B, C, D, W[61] );
	P( D, E, A, B, C, W[62] );
	P( C, D, E, A, B, W[63] );
	P( B, C, D, E, A, W[64] );
	P( A, B, C, D, E, W[65] );
	P( E, A, B, C, D, W[66] );
	P( D, E, A, B, C, W[67] );
	P( C, D, E, A, B, W[68] );
	P( B, C, D, E, A, W[69] );
	P( A, B, C, D, E, W[70] );
	P( E, A, B, C, D, W[71] );
	P( D, E, A, B, C, W[72] );
	P( C, D, E, A, B, W[73] );
	P( B, C, D, E, A, W[74] );
	P( A, B, C, D, E, W[75] );
	P( E, A, B, C, D, W[76] );
	P( D, E, A, B, C, W[77] );
	P( C, D, E, A, B, W[78] );
	P( B, C, D, E, A, W[79] );
  
#undef K
#undef F

	tmp->state[0] += A;
	tmp->state[1] += B;
	tmp->state[2] += C;
	tmp->state[3] += D;
	tmp->state[4] += E;
	
}
__global__ void sha1_kernel_global (const unsigned char *data, sha1_gpu_context *ctx, unsigned int *extended)
{
	int thread_index = threadIdx.x + blockDim.x * blockIdx.x;
	int total_threads = blockDim.x * gridDim.x;
	unsigned int temp, t;
	unsigned int *data2 = (unsigned int*)data;
	/*
	 * Extend 32 block byte block into 80 byte block.
	 */
	for (int i = thread_index; i < dblocks; i += total_threads)
	{	
		unsigned int index_ext = i * 80;
		unsigned int index_data = i * 16;
		GET_UINT32_BE( extended[index_ext    ], data2[index_data]);
		GET_UINT32_BE( extended[index_ext + 1], data2[index_data + 1]);
		GET_UINT32_BE( extended[index_ext + 2], data2[index_data + 2]);
		GET_UINT32_BE( extended[index_ext + 3], data2[index_data + 3]);
		GET_UINT32_BE( extended[index_ext + 4], data2[index_data + 4]);
		GET_UINT32_BE( extended[index_ext + 5], data2[index_data + 5]);
		GET_UINT32_BE( extended[index_ext + 6], data2[index_data + 6]);
		GET_UINT32_BE( extended[index_ext + 7], data2[index_data + 7]);
		GET_UINT32_BE( extended[index_ext + 8], data2[index_data + 8]);
		GET_UINT32_BE( extended[index_ext + 9], data2[index_data + 9]);
		GET_UINT32_BE( extended[index_ext +10], data2[index_data + 10] );
		GET_UINT32_BE( extended[index_ext +11], data2[index_data + 11] );
		GET_UINT32_BE( extended[index_ext +12], data2[index_data + 12] );
		GET_UINT32_BE( extended[index_ext +13], data2[index_data + 13] );
		GET_UINT32_BE( extended[index_ext +14], data2[index_data + 14] );
		GET_UINT32_BE( extended[index_ext +15], data2[index_data + 15] );

		for (t = 16; t < 80; t++) {
			temp = extended[index_ext + t - 3] ^ extended[index_ext + t - 8] ^ extended[index_ext + t - 14] ^ extended[index_ext + t - 16];
			extended[index_ext + t] = S(temp,1);
		}

	}
	__syncthreads();
	
	if (thread_index == total_threads - 1) {
		__shared__ sha1_gpu_context tmp;
		tmp.state[0] = 0x67452301;
		tmp.state[1] = 0xEFCDAB89;
		tmp.state[2] = 0x98BADCFE;
		tmp.state[3] = 0x10325476;
		tmp.state[4] = 0xC3D2E1F0;
		for (t = 0; t < dblocks; t++)
			sha1_gpu_process1 (&tmp, (unsigned int*)&extended[t * 80]);
		ctx->state[0] = tmp.state[0];
		ctx->state[1] = tmp.state[1];
		ctx->state[2] = tmp.state[2];
		ctx->state[3] = tmp.state[3];
		ctx->state[4] = tmp.state[4];
	}	
}
/*
 * Run sha1 kernel on GPU
 * input - message
 * size - message size
 * output - buffer to store hash value
 * proc - maximum threads per block
 */
#define HANDLE 16
void sha1_gpu_global (unsigned char *input, unsigned long size, unsigned char *output, int proc)
{
	int total_threads;		/* Total number of threads in the grid */
	int threads_per_block;		/* Number of threads in a block */
	int pad, size_be;		/* Number of zeros to pad, message size in big-enadian. */
	int total_datablocks;		/* Total number of blocks message is split into */
	unsigned int *d_extended[HANDLE];	/* Extended blocks on the device */
	sha1_gpu_context ctx, *d_ctx[HANDLE];	/* Intermediate hash states */
	unsigned int* h_extended[HANDLE];
	/* Initialization vector for SHA-1 */
	ctx.state[0] = 0x67452301;
	ctx.state[1] = 0xEFCDAB89;
	ctx.state[2] = 0x98BADCFE;
	ctx.state[3] = 0x10325476;
	ctx.state[4] = 0xC3D2E1F0;
	
	pad = padding_256 (size);
	threads_per_block = proc;
	/* How many blocks in the message */
	total_datablocks = (size + pad + 8) / 64;          //64byte

	if (total_datablocks > threads_per_block)
		total_threads = threads_per_block;
	else
		total_threads = total_datablocks;
	size_be = LETOBE32 (size * 8);  //how many bits in data
	/* Allocate enough memory on the device */
	printf("total_datablocks %d\n", total_datablocks);
	cutStartTimer (&chmeter.malloc_timer);
	for (int i = 0; i < HANDLE; ++i)
	{
		cudaMallocManaged ((void**)&d_extended[i], total_datablocks * 80 * sizeof(unsigned int));
		cudaMallocManaged ((void**)&d_ctx[i], sizeof (sha1_gpu_context));
		CUT_SAFE_CALL(cudaGetLastError());
	}
	cudaDeviceSynchronize();
	cutStopTimer (&chmeter.malloc_timer, &time_res.malloc_timer_res);

	/*
	 * Copy the data from host to device and perform padding
	 */
	cutStartTimer (&chmeter.memcpy_h2d_timer);
	memset (input + size, 0x80, 1);
	memset (input + size + 1, 0, pad + 7);
	memcpy (input + size + pad + 4, &size_be, 4);
	cudaMemPrefetchAsync(input, size + pad + 8, 0);
	cutStopTimer (&chmeter.memcpy_h2d_timer, &time_res.memcpy_h2d_timer_res);
	
	CUT_SAFE_CALL(cudaMemcpyToSymbol(dblocks, &total_datablocks, sizeof(int)));
	/*
	 * Run the algorithm
	 */
	cutStartTimer (&chmeter.kernel_timer);
	cudaStream_t stream[HANDLE];
	for (int i = 0; i < HANDLE; ++i)
	{
		cudaStreamCreate(&stream[i]);
	}
	for (int i = 0; i < HANDLE; ++i)
	{
		sha1_kernel_global <<<26, 512, 0, stream[i]>>>(input, d_ctx[i], d_extended[i]);
		CUT_SAFE_CALL(cudaGetLastError());
	}
	// for (int i = 0; i < HANDLE; ++i)
	// {
	// 	cudaMemPrefetchAsync(d_extended[i], total_datablocks * 80 * sizeof(unsigned int), cudaCpuDeviceId, stream[i]);
	// }
	cudaDeviceSynchronize();
	cutStopTimer (&chmeter.kernel_timer, &time_res.kernel_timer_res);
	
	// struct timeval s, e;
	// gettimeofday(&s, NULL);
	// for (int i = 0; i < HANDLE; ++i)
	// {
	// 	unsigned int *tmp = d_extended[i];
	// 	for (int t = 0; t < total_datablocks; t++){
	// 		sha1_gpu_process1 (&ctx, &tmp[t*80]);
	// 	}
	// }
	// gettimeofday(&e, NULL);
	// double tmp = (double)s.tv_sec*1000 + (double)s.tv_usec / 1000;
	// double retime = (double)e.tv_sec*1000 + (double)e.tv_usec / 1000 - tmp;
	// printf("cpu time %f ms \n", retime);
	
	/* Put the hash value in the users' buffer */
	cutStartTimer (&chmeter.memcpy_d2h_timer);
	for (int i = 0; i < HANDLE; ++i)
	{
		// PUT_UINT32_BE( ctx.state[0], output,  0 );
		// PUT_UINT32_BE( ctx.state[1], output,  4 );
		// PUT_UINT32_BE( ctx.state[2], output,  8 );
		// PUT_UINT32_BE( ctx.state[3], output, 12 );
		// PUT_UINT32_BE( ctx.state[4], output, 16 );
		PUT_UINT32_BE( d_ctx[i]->state[0], output,  0 );
		PUT_UINT32_BE( d_ctx[i]->state[1], output,  4 );
		PUT_UINT32_BE( d_ctx[i]->state[2], output,  8 );
		PUT_UINT32_BE( d_ctx[i]->state[3], output, 12 );
		PUT_UINT32_BE( d_ctx[i]->state[4], output, 16 );
	}
	cutStopTimer (&chmeter.memcpy_d2h_timer, &time_res.memcpy_d2h_timer_res);

	cutStartTimer (&chmeter.free_timer);
	
	for (int i = 0; i < HANDLE; ++i)
	{
		cudaFree (d_ctx[i]);
		cudaFree (d_extended[i]);
	}
	cutStopTimer (&chmeter.free_timer, &time_res.free_timer_res);
	cudaProfilerStop();
}


int main(int argc, char *argv[])
{
	testvector tv1 = {
		(unsigned char *) "abc",
		(unsigned char *) "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d"
	};				
	testvector tv2 = {
		(unsigned char *) "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
		(unsigned char *) "\x84\x98\x3e\x44\x1c\x3b\xd2\x6e\xba\xae\x4a\xa1\xf9\x51\x29\xe5\xe5\x46\x70\xf1"
	};					   
	unsigned char hash[20];
	unsigned char *data = NULL;
	int max_threads_per_block = MAX_THREADS_PER_BLOCK;

	printf ("===================================\n");
	printf ("SHA-1 HASH ALGORITHM BENCHMARK TEST\n");
	printf ("===================================\n");

	int n_1MB = 1000 * 1000;

	FILE *finput;
	finput = fopen("../data_10MB","rb");
	if(!finput) {
		printf("Unable to open input file\n"); 
		exit(1);
	}
	unsigned long start = ftell(finput);
	fseek(finput, 0L, SEEK_END);
	unsigned long end = ftell(finput);
	fseek(finput, start, SEEK_SET);
	unsigned long flength = end - start;
	cudaMallocManaged((void**)&data, flength * sizeof(char) + padding_256(flength) + 8);
	//data = (unsigned char *)malloc(flength * sizeof(char) + padding_256(flength) + 8);
	if (data == NULL) {
		printf ("ERROR: Insufficient memory on host\n");
		return -1;
	}
	unsigned long rlength= fread(data, 1, flength, finput);
	printf("read length %lu, %lu\n", rlength, flength);
	fclose(finput);

	cutStartTimer(&chmeter.total_timer);
	sha1_gpu_global (data, flength, hash, max_threads_per_block);
	cudaFree(data);
	cutStopTimer(&chmeter.total_timer, &time_res.total_timer_res);
	printf ("GPU hash_size:%d MB\nkernel_time:%f ms\nmemcpy_h2d_time:%f ms\nmemcpy_d2h_time:%f ms\nmalloc_time:%f ms\nfree_time:%f ms\ntotal_time:%f ms\n\n", flength/n_1MB,
			time_res.kernel_timer_res,
			time_res.memcpy_h2d_timer_res,
			time_res.memcpy_d2h_timer_res,
			time_res.malloc_timer_res,
			time_res.free_timer_res,
			time_res.total_timer_res);

	for (int i = 0; i < 20; ++i)
	{
		printf("%x ", hash[i]);
	}
	return 0;
}
