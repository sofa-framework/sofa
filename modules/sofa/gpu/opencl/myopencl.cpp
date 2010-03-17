#include "myopencl.h"
#include <CL/cl.h>
#include <iostream>

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
cl_context _context = NULL;
cl_command_queue* _queues = NULL;
cl_device_id* _devices = NULL;
cl_int _error=CL_SUCCESS;

//private functions


//
// bool addQueue(cl_command_queue queue)
// {
// 	int i;
//
// 	if(_numDevices >= MAX_DEVICES)return false;
//
// 	_numDevices++;
// 	cl_command_queue *q = new cl_command_queue[_numDevices];
//
// 	for(i=0;i<_numDevices-1;i++)q[i]=_queues[i];
// 	q[i]=queue;
//
// 	delete(_queues);
// 	_queues=q;
// 	return true;
// }
//
// cl_command_queue queue(int i){return _queues[i];}
//
// cl_device_id device(int i){return _devices[i];}
//
// cl_context context()
// {
// 	return _context;
// }
//

void releaseContext()
{
    if(_context)clReleaseContext(_context);
}


void releaseQueues()
{
    for(int i=0; i<_numDevices; i++)
        if(_queues[i])clReleaseCommandQueue(_queues[i]);
}

void releaseDevices()
{
    if(_devices)delete(_devices);;
}


cl_context createContext(cl_device_type type)
{
    if(_context)clReleaseContext(_context);
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

}

void createQueues()
{
    if(_queues)releaseQueues();
    _queues = new cl_command_queue[_numDevices];
    for(int i=0; i<_numDevices; i++)
        _queues[i] = clCreateCommandQueue(_context, _devices[i], 0, NULL);
}

//opencl public functions

int myopenclInit(int /*device*/)
{
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

void myopenclCreateBuffer(int /*device*/,void ** dptr,int n)
{
    *dptr = clCreateBuffer(_context,CL_MEM_READ_WRITE,n,NULL,&_error);
}

void myopenclReleaseBuffer(int /*device*/,void * p)
{
    _error = clReleaseMemObject((cl_mem) p);
}

void myopenclEnqueueWriteBuffer(int device,void * ddest,const void* hsrc,size_t n)
{
    _error = clEnqueueWriteBuffer(_queues[device], (cl_mem) ddest, CL_TRUE, 0, n, hsrc,0,NULL,NULL);
}


void myopenclEnqueueReadBuffer(int device,void* hdest,const void * dsrc, size_t n)
{
    _error = clEnqueueReadBuffer(_queues[device], (cl_mem) dsrc, CL_TRUE, 0, n,hdest,0,NULL,NULL);
}

void myopenclEnqueueCopyBuffer(int device, void* ddest,const void * dsrc, size_t n)
{
    _error = clEnqueueCopyBuffer(_queues[device],(cl_mem)ddest,(cl_mem)dsrc,0,0, n,0,NULL,NULL);
}

cl_program myopenclProgramWithSource(std::string &s)
{
    const char* ps = s.c_str();
    const size_t pz = s.size();
    return clCreateProgramWithSource(_context, 1, &ps, &pz, &_error);
}


// information public functions

int myopenclNumDevices()
{
    return _numDevices;
}

//error public functions

cl_int & myopenclError()
{
    return _error;
}

void myopenclShowError(std::string file, int line)
{
    if(_error!=CL_SUCCESS)
    {
        std::cout << "Error (file '" << file << "' line " << line << "): " << _error << std::endl;
    }
}

#if defined(__cplusplus)
}
}
}
#endif
