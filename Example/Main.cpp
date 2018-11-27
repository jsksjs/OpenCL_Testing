#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#pragma comment(lib, "OpenCL.lib")
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#pragma warning(disable: 4996)
#define MAX_srcSize (0x100000)

std::ofstream out;

// for sum
const int LIST_SIZE = 11444777;

// for matrix
#define WA 1024
#define HA 1024
#define WB 1024
#define HB WA
#define WC WB
#define HC HA

// vector sums
void sum(bool GPU) {
	if (GPU) {
		out.open("sumGPU.txt");
		out << "GPU Vector Summation\n";
	}
	else {
		out.open("sumCPU.txt");
		out << "CPU Vector Summation\n";
	}

	cl_bool foundDevice = false;
	cl_uint platformNum = 0, deviceNum = 0;

	cl_kernel kernel;
	cl_int* A;
	cl_int* B;
	cl_int* C;

	FILE* fp;
	char* srcStr;
	size_t srcSize;

	cl_device_type type;
	size_t typeSize;

	cl_device_id dID = NULL;
	cl_uint numDevices;
	cl_uint numPlatforms;

	cl_platform_id* platforms = NULL;
	cl_device_id* devices = NULL;

	cl_context context;

	cl_command_queue cmdQueue;

	cl_mem buffA;
	cl_mem buffB;
	cl_mem buffC;

	cl_program program;
	size_t localItemSize = 256;
	size_t globalItemSize = LIST_SIZE;


	// Create the input vectors
	cl_int i;
	A = (cl_int*)malloc(sizeof(cl_int)*globalItemSize);
	B = (cl_int*)malloc(sizeof(cl_int)*globalItemSize);
	C = (cl_int*)malloc(sizeof(cl_int)*globalItemSize);
	for (i = 0; i < LIST_SIZE; i++) {
		A[i] = i;
		B[i] = LIST_SIZE - i;
	}

	fp = fopen("kernels/Sum.cl", "r");
	if (!fp) {
		out << "Failed to load kernel.\n";		
		exit(1);
	}
	srcStr = (char*)malloc(MAX_srcSize);
	srcSize = fread(srcStr, 1, MAX_srcSize, fp);
	fclose(fp);

	// Get platform and device information
	clGetPlatformIDs(0, NULL, &numPlatforms);
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	clGetPlatformIDs(numPlatforms, platforms, NULL);

	while (!foundDevice && platformNum < numPlatforms) {
		clGetDeviceIDs(platforms[platformNum], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
		clGetDeviceIDs(platforms[platformNum], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
		while (!foundDevice && deviceNum < numDevices) {
			clGetDeviceInfo(devices[deviceNum], CL_DEVICE_TYPE, 0, NULL, &typeSize);
			type = (cl_device_type)malloc(typeSize);
			clGetDeviceInfo(devices[deviceNum], CL_DEVICE_TYPE, typeSize, &type, NULL);
			if (GPU && (cl_device_type)type == CL_DEVICE_TYPE_GPU) {
				dID = devices[deviceNum];
				foundDevice = true;
			}
			else if (!GPU && (cl_device_type)type == CL_DEVICE_TYPE_CPU) {
				dID = devices[deviceNum];
				foundDevice = true;
			}
			deviceNum++;
		}
		deviceNum = 0;
		platformNum++;
	}
	if (!foundDevice) {
		exit(-1);
	}
	free(devices);
	free(platforms);

	// device name and version
	char name[1024];
	clGetDeviceInfo(dID, CL_DEVICE_NAME, sizeof(name), name, NULL);
	out << name << "\n";
	char ver[1024];
	clGetDeviceInfo(dID, CL_DEVICE_VERSION, sizeof(ver), ver, NULL);
	out << ver << "\n";

	// Create an OpenCL context
	context = clCreateContext(0, 1, &dID, NULL, NULL, NULL);

	// Create a command queue
	cmdQueue = clCreateCommandQueue(context, dID, 0, NULL);

	// Create memory buffers on the device for each vector 
	buffA = clCreateBuffer(context, CL_MEM_READ_ONLY,
		globalItemSize * sizeof(cl_int), NULL, NULL);
	buffB = clCreateBuffer(context, CL_MEM_READ_ONLY,
		globalItemSize * sizeof(cl_int), NULL, NULL);
	buffC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		globalItemSize * sizeof(cl_int), NULL, NULL);

	// Create a program from the kernel source
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr, (const size_t *)&srcSize, NULL);
#ifdef MAC
	const char* flags = "-cl-fast-relaxed-math -DMAC";
#else
	const char* flags = "-cl-fast-relaxed-math";
#endif
	// Build the program
	clBuildProgram(program, 1, &dID, NULL, NULL, NULL);

	// Create the OpenCL kernel
	kernel = clCreateKernel(program, "Add", NULL);

	// Set the arguments of the kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffB);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffC);

	// Copy the lists A and B to their respective memory buffers
	clEnqueueWriteBuffer(cmdQueue, buffA, CL_FALSE, 0,
		globalItemSize * sizeof(cl_int), A, 0, NULL, NULL);

	clEnqueueWriteBuffer(cmdQueue, buffB, CL_FALSE, 0,
		globalItemSize * sizeof(cl_int), B, 0, NULL, NULL);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	// Execute the OpenCL kernel on the list
	clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL,
		&globalItemSize, &localItemSize, 0, NULL, NULL);

	clFinish(cmdQueue);

	// ...
	// do some busy work
	// ...

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	out << duration << " microsecs";

	// Read the memory buffer C on the device to the local variable C
	//clEnqueueReadBuffer(cmdQueue, buffC, CL_TRUE, 0,
		//globalItemSize * sizeof(cl_int), C, 0, NULL, NULL);

	// release OpenCL
	clReleaseMemObject(buffA);
	clReleaseMemObject(buffB);
	clReleaseMemObject(buffC);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);

	//release host memory
	free(A);
	free(B);
	free(C);

	out.close();
}
void noCL() {
	out.open("noCL.txt");
	out << "NoCL Vector Summation\n";

	// Create the input vectors
	int i;
	int* A = new int[LIST_SIZE];
	int* B = new int[LIST_SIZE];
	int* C = new int[LIST_SIZE];
	for (i = 0; i < LIST_SIZE; i++) {
		A[i] = i;
		B[i] = LIST_SIZE - i;
	}
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < LIST_SIZE; i++) {
		C[i] = A[i] + B[i];
	}
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	out << duration << " microsecs";
	out.close();
}
// Input random float entries into matrix
void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}
// matrix multiplication
void mul(bool GPU) {
	if (GPU) {
		out.open("mulGPU.txt");
		out << "GPU Matrix Multiplication\n";
	}
	else {
		out.open("mulCPU.txt");
		out << "CPU Matrix Multiplication\n";
	}

	srand(3210);

	cl_kernel kernel;


	cl_bool foundDevice = false;
	cl_uint platformNum = 0, deviceNum = 0;

	// host mem alloc for A and B matrices
	unsigned int sizeA = WA * HA;
	unsigned int memA = sizeof(float) * sizeA;
	float* hA = (float*)malloc(memA);

	unsigned int sizeB = WB * HB;
	unsigned int memB = sizeof(float) * sizeB;
	float* hB = (float*)malloc(memB);

	// host mem init
	randomInit(hA, sizeA);
	randomInit(hB, sizeB);

	// host mem alloc for C result
	unsigned int sizeC = WC * HC;
	unsigned int memC = sizeof(float) * sizeC;
	float* hC = (float*)malloc(memC);

	FILE* fp;

	cl_device_id dID = NULL;
	cl_uint numDevices;
	cl_uint numPlatforms;

	cl_platform_id* platforms = NULL;
	cl_device_id* devices = NULL;

	cl_context context;

	cl_command_queue cmdQueue;


	cl_device_type type;
	size_t typeSize;

	cl_mem buffA;
	cl_mem buffB;
	cl_mem buffC;

	cl_program program;
	size_t globalItemSize[2], localItemSize[2], srcSize;
	char* srcStr;

	cl_int err;

	fp = fopen("kernels/Mul.cl", "r");
	if (!fp) {
		exit(1);
	}
	srcStr = (char*)malloc(MAX_srcSize);
	srcSize = fread(srcStr, 1, MAX_srcSize, fp);
	fclose(fp);


	// Get platform and device information
	clGetPlatformIDs(0, NULL, &numPlatforms);
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

	clGetPlatformIDs(numPlatforms, platforms, NULL);

	while (!foundDevice && platformNum < numPlatforms) {
		clGetDeviceIDs(platforms[platformNum], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
		clGetDeviceIDs(platforms[platformNum], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
		while (!foundDevice && deviceNum < numDevices) {
			clGetDeviceInfo(devices[deviceNum], CL_DEVICE_TYPE, 0, NULL, &typeSize);
			type = (cl_device_type)malloc(typeSize);
			clGetDeviceInfo(devices[deviceNum], CL_DEVICE_TYPE, typeSize, &type, NULL);
			if (GPU && (cl_device_type)type == CL_DEVICE_TYPE_GPU) {
				dID = devices[deviceNum];
				foundDevice = true;
			}
			else if (!GPU && (cl_device_type)type == CL_DEVICE_TYPE_CPU) {
				dID = devices[deviceNum];
				foundDevice = true;
			}
			deviceNum++;
		}
		deviceNum = 0;
		platformNum++;
	}
	if (!foundDevice) {
		exit(-1);
	}
	free(devices);
	free(platforms);
	// device name and version
	char name[1024];
	clGetDeviceInfo(dID, CL_DEVICE_NAME, sizeof(name), name, NULL);
	out << name << "\n";
	char ver[1024];
	clGetDeviceInfo(dID, CL_DEVICE_VERSION, sizeof(ver), ver, NULL);
	out << ver << "\n";

	// print matrices
	out << "\n\nA\n";
	for (int i = 0; i < sizeA; i++)
	{
		out << hA[i] << " ";
		if (((i + 1) % WA) == 0)
			out << "\n";
	}
	out << "\n\nB\n";
	for (int i = 0; i < sizeB; i++)
	{
		out << hB[i] << " ";
		if (((i + 1) % WB) == 0)
			out << "\n";
	}
	// Create an OpenCL context
	context = clCreateContext(0, 1, &dID, NULL, NULL, NULL);

	// Create a command queue
	cmdQueue = clCreateCommandQueue(context, dID, 0, NULL);

	// Create memory buffers on the device for each vector 
	buffC = clCreateBuffer(context, CL_MEM_READ_WRITE, memA, NULL, NULL);
	buffA = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		memA, hA, NULL);
	buffB = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		memB, hB, NULL);


	// Create a program from the kernel source
	program = clCreateProgramWithSource(context,
		1, (const char **)&srcStr,
		(const size_t*)&srcSize, &err);

	// Build the program
	err = clBuildProgram(program, 1, &dID, NULL, NULL, NULL);

	// Create the OpenCL kernel
	kernel = clCreateKernel(program, "Mul", &err);

	int wA = WA;
	int wC = WC;

	// Set the arguments of the kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffC);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffA);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffB);
	clSetKernelArg(kernel, 3, sizeof(int), (void*)&wA);
	clSetKernelArg(kernel, 4, sizeof(int), (void*)&wC);

	localItemSize[0] = 16;
	localItemSize[1] = 16;
	globalItemSize[0] = 1024;
	globalItemSize[1] = 1024;

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	// Execute the OpenCL kernel on the list
	err = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL,
		globalItemSize, localItemSize, 0, NULL, NULL);

	// ...
	// optionally do some busy work
	// ...

	clFinish(cmdQueue);

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

	// Read the memory buffer
	err = clEnqueueReadBuffer(cmdQueue, buffC, CL_TRUE, 0,
		memC, hC, 0, NULL, NULL);

	out << "\n\nResult\n";
	for (int i = 0; i < sizeC; i++) {
		out << hC[i] << " ";
		if (((i + 1) % WC) == 0)
			out << "\n";
	}
	out << "\n";

	out << duration << " microsecs";

	// release OpenCL
	clReleaseMemObject(buffA);
	clReleaseMemObject(buffB);
	clReleaseMemObject(buffC);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);

	//release host memory
	free(hA);
	free(hB);
	free(hC);
	out.close();
}

int main(void) {	
	noCL();	
	sum(true);
	sum(false);
	mul(true);
	mul(false);
	return 0;
}