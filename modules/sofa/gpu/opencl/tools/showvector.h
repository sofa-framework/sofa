#ifndef SHOWVECTOR_H
#define SHOWVECTOR_H

#include <stdio.h>
#include <iostream>
#include "../myopencl.h"

class ShowVector
{
    FILE* _file;
    std::string _name;
public:
    ShowVector(char* fileName);
    ~ShowVector();

    template <class T> void writeVector(T v);

    template <class T> void addVector(T* v,int size)
    {
        std::cout << "write in:" << _name << std::endl;
        for(int i=0; i<size; i++)
        {
            writeVector<T>(v[i]);
            if(i%1024==1023)
                fprintf(_file,"\n");
            else fprintf(_file,";");
        }
        fprintf(_file,"\n\n");
    }

    template <class T> void addOpenCLVector(sofa::gpu::opencl::_device_pointer &dp,int size)
    {
        T * tab = new T[size];
        sofa::gpu::opencl::myopenclEnqueueReadBuffer(0,tab,dp.m,dp.offset,size*sizeof(T));
        addVector<T>(tab,size);

        delete(tab);
    }

    template <class T> void addCudaVector(sofa::gpu::opencl::_device_pointer &dp,int size)
    {
        T * tab = new T[size];
        sofa::gpu::opencl::myopenclEnqueueReadBuffer(0,tab,dp.m,dp.offset,size*sizeof(T));
        addVector<T>(tab,size);

        delete(tab);
    }
};

#endif // SHOWVECTOR_H
