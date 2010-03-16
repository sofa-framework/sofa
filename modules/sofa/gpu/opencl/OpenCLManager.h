#ifndef OPENCLMANAGER_H
#define OPENCLMANAGER_H


#include <CL/cl.h>
#include <iostream>
#include <string>

namespace sofa
{

namespace helper
{

class OpenCLManager
{
    static cl_command_queue* _queues;	///< List of command queues
    static cl_context _context;			///< Context used
    static int _numDevices;				///< Devices counting
    static cl_int _error;
    static cl_device_id* _devices;

public:

    enum { MAX_DEVICES = 8 };

    /**
      * Add a command queue to the static command queues list.
      * @param queue command queue to add
      */
    static bool addQueue(cl_command_queue queue)
    {
        int i;

        if(_numDevices >= MAX_DEVICES)return false;

        _numDevices++;
        cl_command_queue *q = new cl_command_queue[_numDevices];

        for(i=0; i<_numDevices-1; i++)q[i]=_queues[i];
        q[i]=queue;

        delete(_queues);
        _queues=q;
        return true;
    }

    static cl_command_queue queue(int i) {return _queues[i];}

    static cl_device_id device(int i) {return _devices[i];}

    static cl_context context()
    {
        return _context;
    }

    static void releaseContext()
    {
        if(context)clReleaseContext(_context);
    }

    static void releaseQueues()
    {
        for(int i=0; i<_numDevices; i++)
            if(queue(i))clReleaseCommandQueue(queue(i));
    }

    static void releaseDevices()
    {
        if(_devices)delete(_devices);;
    }


    static cl_context createContext(cl_device_type type)
    {
        if(_context)clReleaseContext(_context);
        return (_context = clCreateContextFromType(0, type, NULL, NULL, &_error));
    }

    static void createDevices()
    {
        if(_devices)delete(_devices);
        size_t devices_size;
        clGetContextInfo(_context, CL_CONTEXT_DEVICES,0,NULL, &devices_size);		//compter le nombre de matériel
        _numDevices = devices_size/sizeof(cl_device_id);
        _devices = new cl_device_id[_numDevices];					//allouer un espace mémoire pour recueillir les matériels
        clGetContextInfo(_context, CL_CONTEXT_DEVICES,devices_size,_devices, NULL);	//créer une liste de matériel

    }

    static void createQueues()
    {
        if(_queues)releaseQueues();
        _queues = new cl_command_queue[_numDevices];
        for(int i=0; i<_numDevices; i++)
            _queues[i] = clCreateCommandQueue(_context, _devices[i], 0, NULL);
    }


    /**
      * Return devices counting
      */
    static int numDevices()
    {
        return _numDevices;
    }

    static cl_int & error() {return _error;}

    static void showError(std::string file, int line)
    {
        if(error()!=CL_SUCCESS)
        {
            std::cout << "Error (file '" << file << "' line " << line << "): " << error() << std::endl;
        }
    }
};

}

}


#include "OpenCLManager.inl"
#endif // OPENCLMANAGER_H
