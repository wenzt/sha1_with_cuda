/*
 * SHA-1 GPU implementation.
 * 2008, Tadas Vilkeliskis <vilkeliskis.t@gmail.com>
 */
#include <cuda.h>
#include "common.h"
#include <stdio.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
#define S(x,n) ((x << n) | ((x) >> (32 - n)))
#define R(t) \
	temp = extended[block_index + t -  3] ^ extended[block_index + t - 8] ^     \
		   extended[block_index + t - 14] ^ extended[block_index + t - 16]; \
	extended[block_index + t] = S(temp,1); \


typedef struct {
	unsigned int state[5];
} sha1_gpu_context;

/*
 * Process extended block.
 */
__device__ void sha1_gpu_process (sha1_gpu_context *tmp, unsigned int *W)
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

__global__ void sha1_kernel_global1 (const unsigned char *data, sha1_gpu_context *ctx, unsigned int *extended)
{
	int thread_index = threadIdx.x + blockDim.x * blockIdx.x;
	int total_threads = blockDim.x * gridDim.x;
	unsigned int temp, t;
	unsigned int *data2 = (unsigned int*)data;
	/*
	 * Extend 32 block byte block into 80 byte block.
	 */
	for (int i = thread_index; i < 111; i += total_threads)
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
	

	/* Wait for the last thread and compute intermediate hash values of extended blocks */
	// __syncthreads();
	
	// if (thread_index == total_threads - 1) {
	// 	__shared__ sha1_gpu_context tmp;
	// 	tmp.state[0] = 0x67452301;
	// 	tmp.state[1] = 0xEFCDAB89;
	// 	tmp.state[2] = 0x98BADCFE;
	// 	tmp.state[3] = 0x10325476;
	// 	tmp.state[4] = 0xC3D2E1F0;
	// 	for (t = 0; t < block_nums; t++)
	// 		sha1_gpu_process (&tmp, (unsigned int*)&extended[t * 80]);
	// 	ctx->state[0] = tmp.state[0];
	// 	ctx->state[1] = tmp.state[1];
	// 	ctx->state[2] = tmp.state[2];
	// 	ctx->state[3] = tmp.state[3];
	// 	ctx->state[4] = tmp.state[4];
	// }	
}

