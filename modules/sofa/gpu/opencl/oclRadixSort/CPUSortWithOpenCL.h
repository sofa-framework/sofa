#ifndef CPUSORTWITHOPENCL_H
#define CPUSORTWITHOPENCL_H

#include "../myopencl.h"
#include <vector>




template <class T>
class CPUSortWithOpenCL
{
private:

    struct Element
    {
        int key;
        T value;
    };

    typedef std::vector< Element > Vectorsort;

public:
    static void sort(_device_pointer &keys,_device_pointer &values,int numElements)
    {
        T* valueVector = new T[numElements];
        int *keyVector = new int[numElements];
        Vectorsort vecsort;

//		std::cout << "CPUSort values:" <<values.offset << "\nkeys"<< keys.offset <<"\n";

        sofa::gpu::opencl::myopenclEnqueueReadBuffer(0,valueVector,values.m,values.offset,numElements*sizeof(T));
        sofa::gpu::opencl::myopenclEnqueueReadBuffer(0,keyVector,keys.m,keys.offset,numElements*sizeof(int));

        for(int i=0; i<numElements; i++)
        {
            //		float *t = (float*)(valueVector+i);
            //		std::cout << "#" << keyVector[i] << " " << t[0] << "." << t[1] << "." << t[2] << "\n";
            Element e;
            e.key = keyVector[i];
            e.value = valueVector[i];
            vecsort.insert(vecsort.end(),e);
        }

        std::sort( vecsort.begin(), vecsort.end(), compare);

//	std::cout << "-----------------\n";
        for (unsigned int j=0; j<vecsort.size(); j++)
        {
            //		float *t = (float*)(&(vecsort[j].value));
            //		std::cout << " " << vecsort[j].key <<":"<< t[0] << "#" << t[1] << "#" << t[2] << "\n";
            keyVector[j] = vecsort[j].key;
            valueVector[j] = vecsort[j].value;
        }

        sofa::gpu::opencl::myopenclEnqueueWriteBuffer(0,values.m,values.offset,valueVector,numElements*sizeof(T));
        sofa::gpu::opencl::myopenclEnqueueWriteBuffer(0,keys.m,keys.offset,keyVector,numElements*sizeof(int));
    }

private:
    static bool compare(Element e1,Element e2) {return (e1.key<e2.key);}
};

#endif // CPUSORTWITHOPENCL_H
