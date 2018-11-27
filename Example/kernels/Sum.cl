// kernel
__kernel void Add(__global const int* A,
	__global const int* B,
	__global int* C)
{
	// GPUs lend themselves to arrays of vector data 
	// ([a0, a1, a2, a3, ...]+[b0, b1, b2, b3, ...]=[a0+b0, a1+b1, a2+b2, a3+b3, ...]

	// Get the index of the current element to be processed
	int i = get_global_id(0);

	C[i] = A[i] + B[i];

}
