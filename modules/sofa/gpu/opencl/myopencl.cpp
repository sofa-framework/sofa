#include "myopencl.h"

#include <iostream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sofa/helper/BackTrace.h>

#define DEBUG_TEXT(t) //printf("\t  %s\n",t);
#define CL_KERNEL_PATH "/modules/sofa/gpu/opencl/kernels/"

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace opencl
{
#endif



//private data
int _numDevices = 0;
int myopenclMultiOpMax = 0;
cl_context _context = NULL;
cl_command_queue* _queues = NULL;
cl_device_id* _devices = NULL;
cl_int _error=CL_SUCCESS;

std::string _mainPath;

//private functions

void createPath()
{
    _mainPath = getcwd(NULL,0);
    _mainPath += CL_KERNEL_PATH;
}


void releaseContext()
{
    if(_context)clReleaseContext(_context);
    myopenclShowError(__FILE__, __LINE__);
}

void listDevices(cl_device_type type,cl_platform_id platform)
{
    //number of devices on the platform
    cl_uint size_device;
    clGetDeviceIDs(platform,type,0,NULL,&size_device);
    cl_device_id *device = new cl_device_id[size_device];
    clGetDeviceIDs(platform,type,size_device,device,NULL);

    //for each device, display info
    for(cl_uint i=0; i<size_device; i++)
    {
        std::cout << "----------\n";
        size_t size;
        clGetDeviceInfo(device[i],CL_DEVICE_NAME,0,NULL,&size);
        char * data = new char[size];
        clGetDeviceInfo(device[i],CL_DEVICE_NAME,size,data,NULL);
        std::cout << "  DEVICE NAME:    " << data << std::endl;
        delete(data);

        clGetDeviceInfo(device[i],CL_DEVICE_VERSION,0,NULL,&size);
        data = new char[size];
        clGetDeviceInfo(device[i],CL_DEVICE_VERSION,size,data,NULL);
        std::cout << "  DEVICE VERSION: " << data << std::endl;
        delete(data);

        cl_ulong l;
        clGetDeviceInfo(device[i],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),&l,NULL);
        std::cout << "  GLOBAL MEM SIZE: " << l << std::endl;

        clGetDeviceInfo(device[i],CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof(cl_ulong),&l,NULL);
        std::cout << "  GLOBAL MEM CACHE SIZE: " << l << std::endl;

        clGetDeviceInfo(device[i],CL_DEVICE_LOCAL_MEM_SIZE,sizeof(cl_ulong),&l,NULL);
        std::cout << "  LOCAL MEM SIZE: " << l << std::endl;

        cl_uint ui;
        clGetDeviceInfo(device[i],CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(cl_ulong),&ui,NULL);
        std::cout << "  MAX CLOCK FREQUENCY: " << ui << std::endl;

        clGetDeviceInfo(device[i],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_ulong),&ui,NULL);
        std::cout << "  MAX COMPUTE UNITS: " << ui << std::endl;
    }
    delete(device);
}

void listPlatform()
{
    //number of platforms in the computer
    cl_uint size_platform;
    clGetPlatformIDs(0,NULL,&size_platform);
    cl_platform_id * platform = new cl_platform_id();
    clGetPlatformIDs(size_platform,platform,NULL);

    std::cout << std::endl << std::endl << "=======================================" << std::endl << "PLATFORMS LIST"<< std::endl;

    //for each platform, display info and search devices
    for(cl_uint i=0; i<size_platform; i++)
    {
        //display info
        std::cout << "=======================================" << std::endl;
        size_t size;
        // name
        clGetPlatformInfo(platform[i],CL_PLATFORM_NAME,0,NULL,&size);
        char * data = new char[size];
        clGetPlatformInfo(platform[i],CL_PLATFORM_NAME,size,data,NULL);
        std::cout << "PLATFORM NAME:    " << data << std::endl;
        delete(data);
        // version
        clGetPlatformInfo(platform[i],CL_PLATFORM_VERSION,0,NULL,&size);
        data = new char[size];
        clGetPlatformInfo(platform[i],CL_PLATFORM_VERSION,size,data,NULL);
        std::cout << "PLATFORM VERSION: " << data << std::endl;
        delete(data);
        // profile
        clGetPlatformInfo(platform[i],CL_PLATFORM_PROFILE,0,NULL,&size);
        data = new char[size];
        clGetPlatformInfo(platform[i],CL_PLATFORM_PROFILE,size,data,NULL);
        std::cout << "PLATFORM PROFILE: " << data << std::endl;
        delete(data);
        // vendor
        clGetPlatformInfo(platform[i],CL_PLATFORM_VENDOR,0,NULL,&size);
        data = new char[size];
        clGetPlatformInfo(platform[i],CL_PLATFORM_VENDOR,size,data,NULL);
        std::cout << "PLATFORM VENDOR:  " << data << std::endl;
        delete(data);
        // extensions
        clGetPlatformInfo(platform[i],CL_PLATFORM_EXTENSIONS,0,NULL,&size);
        data = new char[size];
        clGetPlatformInfo(platform[i],CL_PLATFORM_EXTENSIONS,size,data,NULL);
        std::cout << "PLATFORM EXTENSI: " << data << std::endl;
        delete(data);

        std::cout << "------------\n CPU\n";
        listDevices(CL_DEVICE_TYPE_CPU,platform[i]);

        std::cout << "------------\n GPU\n";
        listDevices(CL_DEVICE_TYPE_GPU,platform[i]);

        std::cout << "------------\n ACCELERATOR\n";
        listDevices(CL_DEVICE_TYPE_ACCELERATOR,platform[i]);
    }
    std::cout << std::endl << "=======================================" << std::endl;

    delete(platform);
}

void releaseQueues()
{
    for(int i=0; i<_numDevices; i++)
        if(_queues[i])
        {
            clReleaseCommandQueue(_queues[i]);
            myopenclShowError(__FILE__, __LINE__);
        }
}

void releaseDevices()
{
    if(_devices)delete(_devices);;
}


cl_context createContext(cl_device_type type)
{
    listPlatform();
    if(_context)clReleaseContext(_context);
    myopenclShowError(__FILE__, __LINE__);
    return (_context = clCreateContextFromType(0, type, NULL, NULL, &_error));
}

void createDevices()
{
    if(_devices)delete(_devices);
    size_t devices_size;
    clGetContextInfo(_context, CL_CONTEXT_DEVICES,0,NULL, &devices_size);		//compter le nombre de matériel
    _numDevices = devices_size/sizeof(cl_device_id);
    _devices = new cl_device_id[_numDevices];					//allouer un espace mémoire pour recueillir les matériels
    clGetContextInfo(_context, CL_CONTEXT_DEVICES,devices_size,_devices, NULL);	//créer une liste de matériel
    myopenclShowError(__FILE__, __LINE__);

}

void createQueues()
{
    if(_queues)releaseQueues();
    _queues = new cl_command_queue[_numDevices];
    for(int i=0; i<_numDevices; i++)
    {
        _queues[i] = clCreateCommandQueue(_context, _devices[i], 0, NULL);
        myopenclShowError(__FILE__, __LINE__);
    }
}

//opencl public functions

int myopenclInit(int /*device*/)
{
    DEBUG_TEXT("myopenclInit");
    createPath();
    createContext(CL_DEVICE_TYPE_GPU);
    createDevices();
    createQueues();
    if(_error==CL_SUCCESS)return 1;
    else return 0;
}

int myopenclClose()
{
    releaseQueues();
    releaseDevices();
    releaseQueues();
    if(_error==CL_SUCCESS)return 1;
    else return 0;
}

int myopenclGetnumDevices()
{
    return _numDevices;
}

void myopenclCreateBuffer(int /*device*/,cl_mem* dptr,int n)
{
    DEBUG_TEXT("myopenclCreateBuffer ");
    *dptr = clCreateBuffer(_context,CL_MEM_READ_WRITE,n,NULL,&_error);
    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclCreateBuffer ");
}

void myopenclReleaseBuffer(int /*device*/,cl_mem p)
{
    DEBUG_TEXT("myopenclReleaseBuffer ");
    _error = clReleaseMemObject((cl_mem) p);
    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclReleaseBuffer ");
}

void myopenclEnqueueWriteBuffer(int device,cl_mem ddest,size_t offset,const void* hsrc,size_t n)
{
    DEBUG_TEXT("myopenclEnqueueWriteBuffer");
    _error = clEnqueueWriteBuffer(_queues[device], ddest, CL_TRUE, offset, n, hsrc,0,NULL,NULL);
    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclEnqueueWriteBuffer");
}


void myopenclEnqueueReadBuffer(int device,void* hdest,const cl_mem dsrc,size_t offset, size_t n)
{
    DEBUG_TEXT("myopenclEnqueueReadBuffer");
    _error = clEnqueueReadBuffer(_queues[device],  dsrc, CL_TRUE, offset, n,hdest,0,NULL,NULL);
    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclEnqueueReadBuffer");
}

void myopenclEnqueueCopyBuffer(int device, cl_mem ddest,size_t destOffset,const cl_mem dsrc,size_t srcOffset, size_t n)
{
    DEBUG_TEXT("myopenclEnqueueCopyBuffer");
    _error = clEnqueueCopyBuffer(_queues[device],dsrc,ddest,srcOffset,destOffset, n,0,NULL,NULL);
    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclEnqueueCopyBuffer");
}

cl_program myopenclProgramWithSource(const char * s,const size_t size)
{
    DEBUG_TEXT("myopenclProgramWithSource");
    return clCreateProgramWithSource(_context, 1, &s, &size, &_error);
    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclProgramWithSource");
}

cl_kernel myopenclCreateKernel(void* p,const char * kernel_name)
{
    DEBUG_TEXT("myopenclCreateKernel");
    return clCreateKernel((cl_program)p, kernel_name, &_error);
    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclCreateKernel");
}

void myopenclSetKernelArg(cl_kernel kernel,int num_arg,int size,void* arg)
{
    DEBUG_TEXT("myopenclSetKernelArg");
    _error = clSetKernelArg(kernel, num_arg,size, arg);
    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclSetKernelArg");
}


void myopenclBuildProgram(void * program)
{
    DEBUG_TEXT("myopenclBuildProgram");
    _error = clBuildProgram((cl_program)program,0,NULL,NULL,NULL,NULL);

    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclBuildProgram");
}

void myopenclBuildProgramWithFlags(void * program, char * flags)
{
    DEBUG_TEXT("myopenclBuildProgram");
    _error = clBuildProgram((cl_program)program,0,NULL,flags,NULL,NULL);

    myopenclShowError(__FILE__, __LINE__);
    DEBUG_TEXT("~myopenclBuildProgram");
}

void myopenclExecKernel(int device,cl_kernel kernel,unsigned int work_dim,const size_t *global_work_offset,const size_t *global_work_size,const size_t *local_work_size)
{
    DEBUG_TEXT("myopenclExecKernel");

    _error = clEnqueueNDRangeKernel(_queues[device],kernel,work_dim,global_work_offset,global_work_size,local_work_size,0,NULL,NULL);

    DEBUG_TEXT("~myopenclExecKernel");
}

// information public functions

int myopenclNumDevices()
{
    DEBUG_TEXT("myopenclNumDevices");
    return _numDevices;
}

extern void* myopencldevice(int device)
{
    DEBUG_TEXT("myopencldevice");
    return (void*) _devices[device];
}

//error public functions

cl_int & myopenclError()
{
    return _error;
}

std::string myopenclErrorMsg(cl_int err)
{
    switch(err)
    {
#define SOFA_CL_ERR(e) case e: return #e
        SOFA_CL_ERR(CL_SUCCESS);
        SOFA_CL_ERR(CL_DEVICE_NOT_FOUND);
        SOFA_CL_ERR(CL_DEVICE_NOT_AVAILABLE);
        SOFA_CL_ERR(CL_COMPILER_NOT_AVAILABLE);
        SOFA_CL_ERR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        SOFA_CL_ERR(CL_OUT_OF_RESOURCES);
        SOFA_CL_ERR(CL_OUT_OF_HOST_MEMORY);
        SOFA_CL_ERR(CL_PROFILING_INFO_NOT_AVAILABLE);
        SOFA_CL_ERR(CL_MEM_COPY_OVERLAP);
        SOFA_CL_ERR(CL_IMAGE_FORMAT_MISMATCH);
        SOFA_CL_ERR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        SOFA_CL_ERR(CL_BUILD_PROGRAM_FAILURE);
        SOFA_CL_ERR(CL_MAP_FAILURE);
        SOFA_CL_ERR(CL_INVALID_VALUE);
        SOFA_CL_ERR(CL_INVALID_DEVICE_TYPE);
        SOFA_CL_ERR(CL_INVALID_PLATFORM);
        SOFA_CL_ERR(CL_INVALID_DEVICE);
        SOFA_CL_ERR(CL_INVALID_CONTEXT);
        SOFA_CL_ERR(CL_INVALID_QUEUE_PROPERTIES);
        SOFA_CL_ERR(CL_INVALID_COMMAND_QUEUE);
        SOFA_CL_ERR(CL_INVALID_HOST_PTR);
        SOFA_CL_ERR(CL_INVALID_MEM_OBJECT);
        SOFA_CL_ERR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        SOFA_CL_ERR(CL_INVALID_IMAGE_SIZE);
        SOFA_CL_ERR(CL_INVALID_SAMPLER);
        SOFA_CL_ERR(CL_INVALID_BINARY);
        SOFA_CL_ERR(CL_INVALID_BUILD_OPTIONS);
        SOFA_CL_ERR(CL_INVALID_PROGRAM);
        SOFA_CL_ERR(CL_INVALID_PROGRAM_EXECUTABLE);
        SOFA_CL_ERR(CL_INVALID_KERNEL_NAME);
        SOFA_CL_ERR(CL_INVALID_KERNEL_DEFINITION);
        SOFA_CL_ERR(CL_INVALID_KERNEL);
        SOFA_CL_ERR(CL_INVALID_ARG_INDEX);
        SOFA_CL_ERR(CL_INVALID_ARG_VALUE);
        SOFA_CL_ERR(CL_INVALID_ARG_SIZE);
        SOFA_CL_ERR(CL_INVALID_KERNEL_ARGS);
        SOFA_CL_ERR(CL_INVALID_WORK_DIMENSION);
        SOFA_CL_ERR(CL_INVALID_WORK_GROUP_SIZE);
        SOFA_CL_ERR(CL_INVALID_WORK_ITEM_SIZE);
        SOFA_CL_ERR(CL_INVALID_GLOBAL_OFFSET);
        SOFA_CL_ERR(CL_INVALID_EVENT_WAIT_LIST);
        SOFA_CL_ERR(CL_INVALID_EVENT);
        SOFA_CL_ERR(CL_INVALID_OPERATION);
        SOFA_CL_ERR(CL_INVALID_GL_OBJECT);
        SOFA_CL_ERR(CL_INVALID_BUFFER_SIZE);
        SOFA_CL_ERR(CL_INVALID_MIP_LEVEL);
        SOFA_CL_ERR(CL_INVALID_GLOBAL_WORK_SIZE);
#undef SOFA_CL_ERR
    default:
    {
        std::ostringstream o;
        o << err;
        return o.str();
    }
    }
}

void myopenclShowError(std::string file, int line)
{
    if(_error!=CL_SUCCESS && _error!=1)
    {
        std::cout << "Error (file '" << file << "' line " << line << "): " << myopenclErrorMsg(_error) << std::endl;
        sofa::helper::BackTrace::dump();
        exit(1);
    }
}

std::string myopenclPath()
{
    return _mainPath;
}

void myopenclBarrier(_device_pointer m, std::string file, int line)
{
    std::cout << file << " " << line << "\n";
    std::cout << "myopenclbarrier-------------------------------------------------\n";
    char p[1];
    myopenclEnqueueReadBuffer(0,p,m.m,0, 1);
    std::cout <<"~myopenclbarrier-------------------------------------------------\n";
}








_device_pointer deviceTmpArray;
size_t TmpArraySize = 0;
int valueTmpArray =0;
void* hostTmpArray = NULL;

void myopenclMemsetDevice(int d, _device_pointer dDestPointer, int value, size_t n)
{
    DEBUG_TEXT("myopenclMemsetDevice");
    if(TmpArraySize<n || value != valueTmpArray)
    {
        DEBUG_TEXT("myopenclMemsetDevice1");
        if(deviceTmpArray.m!=NULL)  myopenclReleaseBuffer(d,deviceTmpArray.m);
        if(hostTmpArray!=NULL)free(hostTmpArray) ;
        myopenclCreateBuffer(d,&(deviceTmpArray.m),n);
        deviceTmpArray.offset=0;
        hostTmpArray = malloc(n);
        memset((void*)hostTmpArray, value, n);
        myopenclEnqueueWriteBuffer(d, deviceTmpArray.m,0, hostTmpArray,n);
        TmpArraySize = n;
        valueTmpArray = value;
    }
    myopenclEnqueueCopyBuffer(d, dDestPointer.m, dDestPointer.offset, deviceTmpArray.m,0, n);

}



















#if defined(__cplusplus)
}
}
}
#endif

#undef DEBUG_TEXT
