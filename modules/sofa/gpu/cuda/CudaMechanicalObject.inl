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
    void MechanicalObjectCudaVec3f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    int MechanicalObjectCudaVec3f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
    void MechanicalObjectCudaVec3f1_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec3f1_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec3f1_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f1_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f1_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec3f1_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    void MechanicalObjectCudaVec3f1_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    int MechanicalObjectCudaVec3f1_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec3f1_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
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
void MechanicalObject<CudaVec3f1Types>::accumulateForce()
{
    if (!this->externalForces->empty())
    {
        gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(this->externalForces->size(), this->f->deviceWrite(), this->externalForces->deviceRead());
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
void MechanicalObject<CudaVec3f1Types>::vAlloc(VecId v)
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
void MechanicalObject<CudaVec3f1Types>::vOp(VecId v, VecId a, VecId b, double f)
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
                gpu::cuda::MechanicalObjectCudaVec3f1_vClear(vv->size(), vv->deviceWrite());
            }
            else
            {
                VecDeriv* vv = getVecDeriv(v.index);
                vv->fastResize(this->vsize);
                gpu::cuda::MechanicalObjectCudaVec3f1_vClear(vv->size(), vv->deviceWrite());
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
                    gpu::cuda::MechanicalObjectCudaVec3f1_vMEq(vv->size(), vv->deviceWrite(), (Real) f);
                }
                else
                {
                    VecDeriv* vv = getVecDeriv(v.index);
                    gpu::cuda::MechanicalObjectCudaVec3f1_vMEq(vv->size(), vv->deviceWrite(), (Real) f);
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
                    gpu::cuda::MechanicalObjectCudaVec3f1_vEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
                }
                else
                {
                    VecDeriv* vv = getVecDeriv(v.index);
                    VecDeriv* vb = getVecDeriv(b.index);
                    vv->fastResize(vb->size());
                    gpu::cuda::MechanicalObjectCudaVec3f1_vEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
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
                gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(vv->size(), vv->deviceWrite(), va->deviceRead());
            }
            else
            {
                VecDeriv* vv = getVecDeriv(v.index);
                VecDeriv* va = getVecDeriv(a.index);
                vv->fastResize(va->size());
                gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(vv->size(), vv->deviceWrite(), va->deviceRead());
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
                            gpu::cuda::MechanicalObjectCudaVec3f1_vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            vv->resize(vb->size());
                            gpu::cuda::MechanicalObjectCudaVec3f1_vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->resize(vb->size());
                        gpu::cuda::MechanicalObjectCudaVec3f1_vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
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
                            gpu::cuda::MechanicalObjectCudaVec3f1_vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            vv->resize(vb->size());
                            gpu::cuda::MechanicalObjectCudaVec3f1_vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->resize(vb->size());
                        gpu::cuda::MechanicalObjectCudaVec3f1_vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
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
                            gpu::cuda::MechanicalObjectCudaVec3f1_vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            gpu::cuda::MechanicalObjectCudaVec3f1_vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* va = getVecDeriv(a.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->fastResize(va->size());
                        gpu::cuda::MechanicalObjectCudaVec3f1_vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
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
                            gpu::cuda::MechanicalObjectCudaVec3f1_vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            gpu::cuda::MechanicalObjectCudaVec3f1_vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* va = getVecDeriv(a.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->fastResize(va->size());
                        gpu::cuda::MechanicalObjectCudaVec3f1_vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
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
void MechanicalObject<gpu::cuda::CudaVec3fTypes>::vMultiOp(const VMultiOp& ops)
{
    // optimize common integration case: v += a*dt, x += v*dt
    if (ops.size() == 2 && ops[0].second.size() == 2 && ops[0].first == ops[0].second[0].first && ops[0].first.type == VecId::V_DERIV && ops[0].second[1].first.type == VecId::V_DERIV
        && ops[1].second.size() == 2 && ops[1].first == ops[1].second[0].first && ops[0].first == ops[1].second[1].first && ops[1].first.type == VecId::V_COORD)
    {
        VecDeriv* va = getVecDeriv(ops[0].second[1].first.index);
        VecDeriv* vv = getVecDeriv(ops[0].first.index);
        VecCoord* vx = getVecCoord(ops[1].first.index);
        const unsigned int n = vx->size();
        const double f_v_v = ops[0].second[0].second;
        const double f_v_a = ops[0].second[1].second;
        const double f_x_x = ops[1].second[0].second;
        const double f_x_v = ops[1].second[1].second;
        gpu::cuda::MechanicalObjectCudaVec3f_vIntegrate(n, va->deviceRead(), vv->deviceWrite(), vx->deviceWrite(), (float)f_v_v, (float)f_v_a, (float)f_x_x, (float)f_x_v);
    }
    else // no optimization for now for other cases
        Inherited::vMultiOp(ops);
}

template <>
void MechanicalObject<gpu::cuda::CudaVec3f1Types>::vMultiOp(const VMultiOp& ops)
{
    // optimize common integration case: v += a*dt, x += v*dt
    if (ops.size() == 2 && ops[0].second.size() == 2 && ops[0].first == ops[0].second[0].first && ops[0].first.type == VecId::V_DERIV && ops[0].second[1].first.type == VecId::V_DERIV
        && ops[1].second.size() == 2 && ops[1].first == ops[1].second[0].first && ops[0].first == ops[1].second[1].first && ops[1].first.type == VecId::V_COORD)
    {
        VecDeriv* va = getVecDeriv(ops[0].second[1].first.index);
        VecDeriv* vv = getVecDeriv(ops[0].first.index);
        VecCoord* vx = getVecCoord(ops[1].first.index);
        const unsigned int n = vx->size();
        const double f_v_v = ops[0].second[0].second;
        const double f_v_a = ops[0].second[1].second;
        const double f_x_x = ops[1].second[0].second;
        const double f_x_v = ops[1].second[1].second;
        gpu::cuda::MechanicalObjectCudaVec3f1_vIntegrate(n, va->deviceRead(), vv->deviceWrite(), vx->deviceWrite(), (float)f_v_v, (float)f_v_a, (float)f_x_x, (float)f_x_v);
    }
    else // no optimization for now for other cases
        Inherited::vMultiOp(ops);
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
double MechanicalObject<CudaVec3f1Types>::vDot(VecId a, VecId b)
{
    Real r = 0.0f;
    if (a.type == VecId::V_COORD && b.type == VecId::V_COORD)
    {
        VecCoord* va = getVecCoord(a.index);
        VecCoord* vb = getVecCoord(b.index);
        int tmpsize = gpu::cuda::MechanicalObjectCudaVec3f1_vDotTmpSize(va->size());
        if (tmpsize == 0)
        {
            gpu::cuda::MechanicalObjectCudaVec3f1_vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), NULL, NULL);
        }
        else
        {
            this->data.tmpdot.fastResize(tmpsize);
            gpu::cuda::MechanicalObjectCudaVec3f1_vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), this->data.tmpdot.deviceWrite(), (float*)(&(this->data.tmpdot.getCached(0))));
        }
    }
    else if (a.type == VecId::V_DERIV && b.type == VecId::V_DERIV)
    {
        VecDeriv* va = getVecDeriv(a.index);
        VecDeriv* vb = getVecDeriv(b.index);
        int tmpsize = gpu::cuda::MechanicalObjectCudaVec3f1_vDotTmpSize(va->size());
        if (tmpsize == 0)
        {
            gpu::cuda::MechanicalObjectCudaVec3f1_vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), NULL, NULL);
        }
        else
        {
            this->data.tmpdot.fastResize(tmpsize);
            gpu::cuda::MechanicalObjectCudaVec3f1_vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), this->data.tmpdot.deviceWrite(), (float*)(&(this->data.tmpdot.getCached(0))));
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
void MechanicalObject<CudaVec3f1Types>::resetForce()
{
    VecDeriv& f= *getF();
    gpu::cuda::MechanicalObjectCudaVec3f1_vClear(f.size(), f.deviceWrite());
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

template <>
void MechanicalObject<CudaVec3f1Types>::getIndicesInSpace(helper::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const
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
