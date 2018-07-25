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

#define MAX_THREADS_PER_BLOCK 128
#define CUT_SAFE_CALL(call)\
    cudaError err = call;\
    if (err != cudaSuccess) {\
        fprintf(stderr, "Cut error in file '%s' in line %i.\n", __FILE__, __LINE__);\
    exit(1);\
}
#pragma pack(4)
typedef struct {
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
//extern void sha1_cpu (unsigned char *input, int ilen, unsigned char *output);
extern __global__ void sha1_kernel_global (const unsigned char  *data, sha1_gpu_context  *ctx, unsigned int  *extended, int block_nums);

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
/*
 * Run sha1 kernel on GPU
 * input - message
 * size - message size
 * output - buffer to store hash value
 * proc - maximum threads per block
 */
void sha1_gpu_global (unsigned char *input, unsigned long size, unsigned char *output, int proc)
{
	int total_threads;		/* Total number of threads in the grid */
	int threads_per_block;		/* Number of threads in a block */
	int pad, size_be;		/* Number of zeros to pad, message size in big-enadian. */
	int total_datablocks;		/* Total number of blocks message is split into */
	unsigned int *d_extended;	/* Extended blocks on the device */
	sha1_gpu_context ctx, *d_ctx;	/* Intermediate hash states */

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
	cutStartTimer (&chmeter.malloc_timer);
	cudaMallocManaged ((void**)&d_extended, total_datablocks * 80 * sizeof(unsigned int));
	cudaMallocManaged ((void**)&d_ctx, sizeof (sha1_gpu_context));
	cutStopTimer (&chmeter.malloc_timer, &time_res.malloc_timer_res);

	/*
	 * Copy the data from host to device and perform padding
	 */
	cutStartTimer (&chmeter.memcpy_h2d_timer);
	//memcpy (d_ctx, &ctx, sizeof (sha1_gpu_context));
	memset (input + size, 0x80, 1);
	memset (input + size + 1, 0, pad + 7);
	memcpy (input + size + pad + 4, &size_be, 4);
	//cudaDeviceSynchronize();
	int device = -1;
  	cudaGetDevice(&device);
  	cudaMemPrefetchAsync(input, size + pad + 8, device, NULL);
	cutStopTimer (&chmeter.memcpy_h2d_timer, &time_res.memcpy_h2d_timer_res);

	/*
	 * Run the algorithm
	 */
	
	sha1_kernel_global <<<1, total_threads>>>(input, d_ctx, d_extended, total_datablocks);
	cudaDeviceSynchronize();
	cutStopTimer (&chmeter.kernel_timer, &time_res.kernel_timer_res);

	cutStartTimer (&chmeter.memcpy_d2h_timer);
	/* Put the hash value in the users' buffer */
	PUT_UINT32_BE( d_ctx->state[0], output,  0 );
	PUT_UINT32_BE( d_ctx->state[1], output,  4 );
	PUT_UINT32_BE( d_ctx->state[2], output,  8 );
	PUT_UINT32_BE( d_ctx->state[3], output, 12 );
	PUT_UINT32_BE( d_ctx->state[4], output, 16 );
	cutStopTimer (&chmeter.memcpy_d2h_timer, &time_res.memcpy_d2h_timer_res);

	cutStartTimer (&chmeter.free_timer);
	cudaFree (d_ctx);
	cudaFree (d_extended);
	cutStopTimer (&chmeter.free_timer, &time_res.free_timer_res);
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

	printf ("\nTesting algorithm correctness...\n");

	cudaMallocManaged((void**)&data, strlen((const char*)tv1.data) + padding_256(strlen((const char*)tv1.data)) + 8);
	memcpy(data, tv1.data, strlen((const char*)tv1.data));
	sha1_gpu_global (data, strlen((const char*)tv1.data), hash, MAX_THREADS_PER_BLOCK);
	if (memcmp (hash, tv1.hash, 20) == 0) printf ("GPU TEST 1 PASSED\n");
	else printf ("GPU TEST 1 FAILED\n");
	cudaFree(data);

	cudaMallocManaged((void**)&data, strlen((const char*)tv2.data) + padding_256(strlen((const char*)tv2.data)) + 8);
	memcpy(data, tv2.data, strlen((const char*)tv2.data));
	sha1_gpu_global (data, strlen((const char*)tv2.data), hash, MAX_THREADS_PER_BLOCK);
	if (memcmp (hash, tv2.hash, 20) == 0) printf ("GPU TEST 2 PASSED\n");
	else printf ("GPU TEST 2 FAILED\n");
	cudaFree(data);

	printf ("Done.\n\n");
	int n_1MB = 1000 * 1000;

	FILE *finput;
	finput = fopen("./cryptopp700.zip","rb");
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
	if (data == NULL) {
		printf ("ERROR: Insufficient memory on host\n");
		return -1;
	}
	unsigned long rlength= fread(data, 1, flength, finput);
	printf("read length %d\n", rlength);

	cutStartTimer(&chmeter.total_timer);
	sha1_gpu_global (data, flength, hash, max_threads_per_block);
	cutStopTimer(&chmeter.total_timer, &time_res.total_timer_res);
	printf ("GPU hash_size:%d MB\nkernel_time:%f ms\nmemcpy_h2d_time:%f ms\nmemcpy_d2h_time:%f ms\nmalloc_time:%f ms\nfree_time:%f ms\ntotal_time:%f ms\n\n", flength/n_1MB,
			time_res.kernel_timer_res,
			time_res.memcpy_h2d_timer_res,
			time_res.memcpy_d2h_timer_res,
			time_res.malloc_timer_res,
			time_res.free_timer_res,
			time_res.total_timer_res);

	cudaFree(data);
	for (int i = 0; i < 20; ++i)
	{
		printf("%x ", hash[i]);
	}
	return 0;
}
