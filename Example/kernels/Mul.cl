// kernel
__kernel void Mul(__global float* C, __global float* A, __global float* B, int wA, int wB)
{
	int tx = get_global_id(0);
	int ty = get_global_id(1);

	// store val
	float value = 0;
	for (int k = 0; k < wA; ++k) {
		float elementA = A[ty * wA + k];
		float elementB = B[k * wB + tx];
		value += elementA * elementB;
	}
	
	// write value
	C[ty * wA + tx] = value;
}