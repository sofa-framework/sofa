#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_INL
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_INL

#include "CudaMechanicalObject.h"
#include <sofa/component/MechanicalObject.inl>


namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void MechanicalObjectCudaVec3f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec3f_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec3f_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec3f_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    int MechanicalObjectCudaVec3f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
}

} // namespace cuda

} // namespace gpu

namespace component
{

using namespace gpu::cuda;

template <>
void MechanicalObject<CudaVec3fTypes>::accumulateForce()
{
    if (!this->externalForces->empty())
    {
        gpu::cuda::MechanicalObjectCudaVec3f_vAssign(this->externalForces->size(), this->f->deviceWrite(), this->externalForces->deviceRead());
    }
}

template <>
void MechanicalObject<CudaVec3fTypes>::vAlloc(VecId v)
{
    if (v.type == VecId::V_COORD && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = getVecCoord(v.index);
        vec->fastResize(vsize);
    }
    else if (v.type == VecId::V_DERIV && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecDeriv* vec = getVecDeriv(v.index);
        vec->fastResize(vsize);
    }
    else
    {
        std::cerr << "Invalid alloc operation ("<<v<<")\n";
        return;
    }
    //vOp(v); // clear vector
}

template <>
void MechanicalObject<CudaVec3fTypes>::vOp(VecId v, VecId a, VecId b, double f)
{
    if(v.isNull())
    {
        // ERROR
        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
        return;
    }
    //std::cout << "> vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
    if (a.isNull())
    {
        if (b.isNull())
        {
            // v = 0
            if (v.type == VecId::V_COORD)
            {
                VecCoord* vv = getVecCoord(v.index);
                vv->fastResize(this->vsize);
                gpu::cuda::MechanicalObjectCudaVec3f_vClear(vv->size(), vv->deviceWrite());
            }
            else
            {
                VecDeriv* vv = getVecDeriv(v.index);
                vv->fastResize(this->vsize);
                gpu::cuda::MechanicalObjectCudaVec3f_vClear(vv->size(), vv->deviceWrite());
            }
        }
        else
        {
            if (b.type != v.type)
            {
                // ERROR
                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                return;
            }
            if (v == b)
            {
                // v *= f
                if (v.type == VecId::V_COORD)
                {
                    VecCoord* vv = getVecCoord(v.index);
                    gpu::cuda::MechanicalObjectCudaVec3f_vMEq(vv->size(), vv->deviceWrite(), (Real) f);
                }
                else
                {
                    VecDeriv* vv = getVecDeriv(v.index);
                    gpu::cuda::MechanicalObjectCudaVec3f_vMEq(vv->size(), vv->deviceWrite(), (Real) f);
                }
            }
            else
            {
                // v = b*f
                if (v.type == VecId::V_COORD)
                {
                    VecCoord* vv = getVecCoord(v.index);
                    VecCoord* vb = getVecCoord(b.index);
                    vv->fastResize(vb->size());
                    gpu::cuda::MechanicalObjectCudaVec3f_vEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
                }
                else
                {
                    VecDeriv* vv = getVecDeriv(v.index);
                    VecDeriv* vb = getVecDeriv(b.index);
                    vv->fastResize(vb->size());
                    gpu::cuda::MechanicalObjectCudaVec3f_vEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
                }
            }
        }
    }
    else
    {
        if (a.type != v.type)
        {
            // ERROR
            std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
            return;
        }
        if (b.isNull())
        {
            // v = a
            if (v.type == VecId::V_COORD)
            {
                VecCoord* vv = getVecCoord(v.index);
                VecCoord* va = getVecCoord(a.index);
                vv->fastResize(va->size());
                gpu::cuda::MechanicalObjectCudaVec3f_vAssign(vv->size(), vv->deviceWrite(), va->deviceRead());
            }
            else
            {
                VecDeriv* vv = getVecDeriv(v.index);
                VecDeriv* va = getVecDeriv(a.index);
                vv->fastResize(va->size());
                gpu::cuda::MechanicalObjectCudaVec3f_vAssign(vv->size(), vv->deviceWrite(), va->deviceRead());
            }
        }
        else
        {
            if (v == a)
            {
                if (f==1.0)
                {
                    // v += b
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        if (b.type == VecId::V_COORD)
                        {
                            VecCoord* vb = getVecCoord(b.index);
                            vv->resize(vb->size());
                            gpu::cuda::MechanicalObjectCudaVec3f_vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            vv->resize(vb->size());
                            gpu::cuda::MechanicalObjectCudaVec3f_vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->resize(vb->size());
                        gpu::cuda::MechanicalObjectCudaVec3f_vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
                else
                {
                    // v += b*f
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        if (b.type == VecId::V_COORD)
                        {
                            VecCoord* vb = getVecCoord(b.index);
                            vv->resize(vb->size());
                            gpu::cuda::MechanicalObjectCudaVec3f_vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            vv->resize(vb->size());
                            gpu::cuda::MechanicalObjectCudaVec3f_vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->resize(vb->size());
                        gpu::cuda::MechanicalObjectCudaVec3f_vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
            }
            else
            {
                if (f==1.0)
                {
                    // v = a+b
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        VecCoord* va = getVecCoord(a.index);
                        vv->fastResize(va->size());
                        if (b.type == VecId::V_COORD)
                        {
                            VecCoord* vb = getVecCoord(b.index);
                            gpu::cuda::MechanicalObjectCudaVec3f_vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            gpu::cuda::MechanicalObjectCudaVec3f_vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* va = getVecDeriv(a.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->fastResize(va->size());
                        gpu::cuda::MechanicalObjectCudaVec3f_vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
                else
                {
                    // v = a+b*f
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        VecCoord* va = getVecCoord(a.index);
                        vv->fastResize(va->size());
                        if (b.type == VecId::V_COORD)
                        {
                            VecCoord* vb = getVecCoord(b.index);
                            gpu::cuda::MechanicalObjectCudaVec3f_vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            gpu::cuda::MechanicalObjectCudaVec3f_vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* va = getVecDeriv(a.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->fastResize(va->size());
                        gpu::cuda::MechanicalObjectCudaVec3f_vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
            }
        }
    }
    //std::cout << "< vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
}

template <>
double MechanicalObject<CudaVec3fTypes>::vDot(VecId a, VecId b)
{
    Real r = 0.0f;
    if (a.type == VecId::V_COORD && b.type == VecId::V_COORD)
    {
        VecCoord* va = getVecCoord(a.index);
        VecCoord* vb = getVecCoord(b.index);
        int tmpsize = gpu::cuda::MechanicalObjectCudaVec3f_vDotTmpSize(va->size());
        if (tmpsize == 0)
        {
            gpu::cuda::MechanicalObjectCudaVec3f_vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), NULL, NULL);
        }
        else
        {
            this->data.tmpdot.fastResize(tmpsize);
            gpu::cuda::MechanicalObjectCudaVec3f_vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), this->data.tmpdot.deviceWrite(), (float*)(&(this->data.tmpdot.getCached(0))));
        }
    }
    else if (a.type == VecId::V_DERIV && b.type == VecId::V_DERIV)
    {
        VecDeriv* va = getVecDeriv(a.index);
        VecDeriv* vb = getVecDeriv(b.index);
        int tmpsize = gpu::cuda::MechanicalObjectCudaVec3f_vDotTmpSize(va->size());
        if (tmpsize == 0)
        {
            gpu::cuda::MechanicalObjectCudaVec3f_vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), NULL, NULL);
        }
        else
        {
            this->data.tmpdot.fastResize(tmpsize);
            gpu::cuda::MechanicalObjectCudaVec3f_vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), this->data.tmpdot.deviceWrite(), (float*)(&(this->data.tmpdot.getCached(0))));
        }
#ifndef NDEBUG
        // Check the result
        //Real r2 = 0.0f;
        //for (unsigned int i=0; i<va->size(); i++)
        //	r2 += (*va)[i] * (*vb)[i];
        //std::cout << "CUDA vDot: GPU="<<r<<"  CPU="<<r2<<" relative error="<<(fabsf(r2)>0.000001?fabsf(r-r2)/fabsf(r2):fabsf(r-r2))<<"\n";
#endif
    }
    else
    {
        std::cerr << "Invalid dot operation ("<<a<<','<<b<<")\n";
    }
    return r;
}

template <>
void MechanicalObject<CudaVec3fTypes>::resetForce()
{
    VecDeriv& f= *getF();
    gpu::cuda::MechanicalObjectCudaVec3f_vClear(f.size(), f.deviceWrite());
}

template <>
void MechanicalObject<CudaVec3fTypes>::getIndicesInSpace(helper::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const
{
    const VecCoord& x = *getX();
    for( unsigned i=0; i<x.size(); ++i )
    {
        if( x[i][0] >= xmin && x[i][0] <= xmax && x[i][1] >= ymin && x[i][1] <= ymax && x[i][2] >= zmin && x[i][2] <= zmax )
        {
            indices.push_back(i);
        }
    }
}

} // namespace component

} // namespace sofa

#endif
