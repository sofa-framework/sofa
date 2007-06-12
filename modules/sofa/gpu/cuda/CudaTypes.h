#ifndef SOFA_GPU_CUDA_CUDATYPES_H
#define SOFA_GPU_CUDA_CUDATYPES_H

//#include "host_runtime.h" // CUDA
#include "CudaCommon.h"
#include "mycuda.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <iostream>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class T>
class CudaVector
{
public:
    typedef T            value_type;
    typedef unsigned int size_type;

protected:
    size_type    vectorSize;     ///< Current size of the vector
    size_type    allocSize;      ///< Allocated size
    void*        devicePointer;  ///< Pointer to the data on the GPU side
    T*           hostPointer;    ///< Pointer to the data on the CPU side
    mutable bool deviceIsValid;  ///< True if the data on the GPU is currently valid
    mutable bool hostIsValid;    ///< True if the data on the CPU is currently valid

public:

    CudaVector()
        : vectorSize(0), allocSize(0), devicePointer(NULL), hostPointer(NULL), deviceIsValid(true), hostIsValid(true)
    {}
    CudaVector(size_type n)
        : vectorSize(0), allocSize(0), devicePointer(NULL), hostPointer(NULL), deviceIsValid(true), hostIsValid(true)
    {
        resize(n);
    }
    CudaVector(const CudaVector<T>& v)
        : vectorSize(0), allocSize(0), devicePointer(NULL), hostPointer(NULL), deviceIsValid(true), hostIsValid(true)
    {
        *this = v;
    }
    void clear()
    {
        vectorSize = 0;
        deviceIsValid = true;
        hostIsValid = true;
    }
    void operator=(const CudaVector<T>& v)
    {
        clear();
        fastResize(v.size());
        deviceIsValid = v.deviceIsValid;
        hostIsValid = v.hostIsValid;
        if (vectorSize!=0 && deviceIsValid)
            mycudaMemcpyDeviceToDevice(devicePointer, v.devicePointer, vectorSize*sizeof(T));
        if (vectorSize!=0 && hostIsValid)
            std::copy(v.hostPointer, v.hostPointer+vectorSize, hostPointer);
    }
    ~CudaVector()
    {
        if (hostPointer!=NULL)
            free(hostPointer);
        if (devicePointer!=NULL)
            mycudaFree(devicePointer);
    }
    size_type size() const
    {
        return vectorSize;
    }
    bool empty() const
    {
        return vectorSize==0;
    }
    void reserve(size_type s)
    {
        if (s <= allocSize) return;
        allocSize = (s>2*allocSize)?s:2*allocSize;
        // always allocate multiples of BSIZE values
        allocSize = (allocSize+BSIZE-1)&-BSIZE;

        void* prevDevicePointer = devicePointer;
        mycudaMalloc(&devicePointer, allocSize*sizeof(T));
        if (vectorSize > 0 && deviceIsValid)
            mycudaMemcpyDeviceToDevice(devicePointer, prevDevicePointer, vectorSize*sizeof(T));
        if (prevDevicePointer != NULL)
            mycudaFree(prevDevicePointer);

        T* prevHostPointer = hostPointer;
        hostPointer = (T*) malloc(allocSize*sizeof(T));
        if (vectorSize!=0 && hostIsValid)
            std::copy(prevHostPointer, prevHostPointer+vectorSize, hostPointer);
        if (prevHostPointer != NULL)
            free(prevHostPointer);
    }
    void resize(size_type s)
    {
        if (s == vectorSize) return;
        reserve(s);
        if (s > vectorSize)
        {
            // Call the constructor for the new elements
            for (size_type i = vectorSize; i < s; i++)
            {
                ::new(hostPointer+i) T;
            }
            if (deviceIsValid)
            {
                if (vectorSize == 0)
                {
                    // wait until the transfer is really necessary, as other modifications might follow
                    deviceIsValid = false;
                }
                else
                {
                    mycudaMemcpyHostToDevice(((T*)devicePointer)+vectorSize, hostPointer+vectorSize, (s-vectorSize)*sizeof(T));
                }
            }
        }
        else
        {
            // Call the destructor for the deleted elements
            for (size_type i = s; i < vectorSize; i++)
            {
                hostPointer[i].~T();
            }
        }
        vectorSize = s;
    }
    void swap(CudaVector<T>& v)
    {
#define VSWAP(type, var) { type t = var; var = v.var; v.var = t; }
        VSWAP(size_type, vectorSize);
        VSWAP(size_type, allocSize);
        VSWAP(void*    , devicePointer);
        VSWAP(T*       , hostPointer);
        VSWAP(bool     , deviceIsValid);
        VSWAP(bool     , hostIsValid);
#undef VSWAP
    }
    const void* deviceRead() const
    {
        copyToDevice();
        return devicePointer;
    }
    void* deviceWrite()
    {
        copyToDevice();
        hostIsValid = false;
        return devicePointer;
    }
    const T* hostRead() const
    {
        copyToHost();
        return hostPointer;
    }
    T* hostWrite()
    {
        copyToHost();
        deviceIsValid = false;
        return hostPointer;
    }
    bool isHostValid() const
    {
        return hostIsValid;
    }
    bool isDeviceValid() const
    {
        return deviceIsValid;
    }
    void push_back(const T& t)
    {
        size_type i = size();
        copyToHost();
        deviceIsValid = false;
        fastResize(i+1);
        ::new(hostPointer+i) T(t);
    }
    void pop_back()
    {
        if (!empty())
            resize(size()-1);
    }
    const T& operator[](size_type i) const
    {
        checkIndex(i);
        return hostRead()[i];
    }
    T& operator[](size_type i)
    {
        checkIndex(i);
        return hostWrite()[i];
    }

    const T& getCached(size_type i) const
    {
        checkIndex(i);
        return hostPointer[i];
    }

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const CudaVector<T>& vec )
    {
        if( vec.size()>0 )
        {
            for( unsigned int i=0; i<vec.size()-1; ++i ) os<<vec[i]<<" ";
            os<<vec[vec.size()-1];
        }
        return os;
    }

    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, CudaVector<T>& vec )
    {
        T t;
        vec.clear();
        while(in>>t)
        {
            vec.push_back(t);
        }
        if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }
    void fastResize(size_type s)
    {
        if (s == vectorSize) return;
        reserve(s);
        vectorSize = s;
    }
protected:
    void copyToHost() const
    {
        if (hostIsValid) return;
#ifndef NDEBUG
        std::cout << "CUDA: GPU->CPU copy of "<<core::objectmodel::Base::decodeTypeName(typeid(*this))<<": "<<vectorSize*sizeof(T)<<" B"<<std::endl;
#endif
        mycudaMemcpyDeviceToHost(hostPointer, devicePointer, vectorSize*sizeof(T));
        hostIsValid = true;
    }
    void copyToDevice() const
    {
        if (deviceIsValid) return;
//#ifndef NDEBUG
        std::cout << "CUDA: CPU->GPU copy of "<<core::objectmodel::Base::decodeTypeName(typeid(*this))<<": "<<vectorSize*sizeof(T)<<" B"<<std::endl;
//#endif
        mycudaMemcpyHostToDevice(devicePointer, hostPointer, vectorSize*sizeof(T));
        deviceIsValid = true;
    }
#ifdef NDEBUG
    void checkIndex(size_type) const
    {
    }
#else
    void checkIndex(size_type i) const
    {
        assert( i<this->size() );
    }
#endif
};

template<class TCoord, class TDeriv, class TReal = typename TCoord::value_type>
class CudaVectorTypes
{
public:
    typedef TCoord Coord;
    typedef TDeriv Deriv;
    typedef TReal Real;
    typedef CudaVector<Coord> VecCoord;
    typedef CudaVector<Deriv> VecDeriv;

    template <class T>
    class SparseData
    {
    public:
        SparseData(unsigned int _index, T& _data): index(_index), data(_data) {};
        unsigned int index;
        T data;
    };

    typedef SparseData<Coord> SparseCoord;
    typedef SparseData<Deriv> SparseDeriv;

    typedef CudaVector<SparseCoord> SparseVecCoord;
    typedef CudaVector<SparseDeriv> SparseVecDeriv;

    //! All the Constraints applied to a state Vector
    typedef	sofa::helper::vector<SparseVecDeriv> VecConst;

    static void set(Coord& c, double x, double y, double z)
    {
        if (c.size()>0)
            c[0] = (typename Coord::value_type)x;
        if (c.size()>1)
            c[1] = (typename Coord::value_type)y;
        if (c.size()>2)
            c[2] = (typename Coord::value_type)z;
    }

    static void get(double& x, double& y, double& z, const Coord& c)
    {
        x = (c.size()>0) ? (double) c[0] : 0.0;
        y = (c.size()>1) ? (double) c[1] : 0.0;
        z = (c.size()>2) ? (double) c[2] : 0.0;
    }

    static void add(Coord& c, double x, double y, double z)
    {
        if (c.size()>0)
            c[0] += (typename Coord::value_type)x;
        if (c.size()>1)
            c[1] += (typename Coord::value_type)y;
        if (c.size()>2)
            c[2] += (typename Coord::value_type)z;
    }

    static const char* Name();
};

typedef sofa::defaulttype::Vec3f Vec3f;
typedef sofa::defaulttype::Vec2f Vec2f;

// GPUs do not support double precision yet
// ( NVIDIA announced at SuperComputing'06 that it will be supported in 2007... )
//typedef sofa::defaulttype::Vec3d Vec3d;
//typedef sofa::defaulttype::Vec2d Vec2d;

//typedef CudaVectorTypes<Vec3d,Vec3d,double> CudaVec3dTypes;
typedef CudaVectorTypes<Vec3f,Vec3f,float> CudaVec3fTypes;
typedef CudaVec3fTypes CudaVec3Types;

template<>
inline const char* CudaVec3fTypes::Name()
{
    return "CudaVec3f";
}

//typedef CudaVectorTypes<Vec2d,Vec2d,double> CudaVec2dTypes;
typedef CudaVectorTypes<Vec2f,Vec2f,float> CudaVec2fTypes;
typedef CudaVec2fTypes CudaVec2Types;

template<>
inline const char* CudaVec2fTypes::Name()
{
    return "CudaVec2f";
}


} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
