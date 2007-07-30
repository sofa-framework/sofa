#ifndef SOFA_GPU_CUDA_CUDARIGIDMAPPING_INL
#define SOFA_GPU_CUDA_CUDARIGIDMAPPING_INL

#include "CudaRigidMapping.h"
#include <sofa/component/mapping/RigidMapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

using sofa::defaulttype::Mat3x3f;
using sofa::defaulttype::Vec3f;

extern "C"
{
    void RigidMappingCuda3f_apply(unsigned int size, const Mat3x3f& rotation, const Vec3f& translation, void* out, void* rotated, const void* in);
    void RigidMappingCuda3f_applyJ(unsigned int size, const Vec3f& v, const Vec3f& omega, void* out, const void* rotated);
    void RigidMappingCuda3f_applyJT(unsigned int size, void* out, const void* rotated, const void* in);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace gpu::cuda;

template <>
void RigidMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    rotatedPoints.resize(points.size());
    out.fastResize(points.size());

    translation = in[index.getValue()].getCenter();
    in[index.getValue()].writeRotationMatrix(rotation);

    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    rotatedPoints[i] = rotation*points[i];
    //    out[i] = rotatedPoints[i];
    //    out[i] += translation;
    //}
    RigidMappingCuda3f_apply(points.size(), rotation, translation, out.deviceWrite(), rotatedPoints.deviceWrite(), points.deviceRead());
}

template <>
void RigidMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    Deriv v,omega;
    out.fastResize(points.size());
    v = in[index.getValue()].getVCenter();
    omega = in[index.getValue()].getVOrientation();
    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    // out = J in
    //    // J = [ I -OM^ ]
    //    out[i] =  v - cross(rotatedPoints[i],omega);
    //}
    RigidMappingCuda3f_applyJ(points.size(), v, omega, out.deviceWrite(), rotatedPoints.deviceRead());
}

template <>
void RigidMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    Deriv v,omega;
    int nbloc = ((points.size()+BSIZE-1)/BSIZE);
    data.tmp.fastResize(2*nbloc);
    RigidMappingCuda3f_applyJT(points.size(), data.tmp.deviceWrite(), rotatedPoints.deviceRead(), in.deviceRead());
    for(int i=0; i<nbloc; i++)
    {
        v += data.tmp[i];
        omega += data.tmp[i+nbloc];
    }
    out[index.getValue()].getVCenter() += v;
    out[index.getValue()].getVOrientation() += omega;
}

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    rotatedPoints.resize(points.size());
    out.fastResize(points.size());

    translation = in[index.getValue()].getCenter();
    in[index.getValue()].writeRotationMatrix(rotation);

    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    rotatedPoints[i] = rotation*points[i];
    //    out[i] = rotatedPoints[i];
    //    out[i] += translation;
    //}
    RigidMappingCuda3f_apply(points.size(), rotation, translation, out.deviceWrite(), rotatedPoints.deviceWrite(), points.deviceRead());
}

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    Deriv v,omega;
    out.fastResize(points.size());
    v = in[index.getValue()].getVCenter();
    omega = in[index.getValue()].getVOrientation();
    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    // out = J in
    //    // J = [ I -OM^ ]
    //    out[i] =  v - cross(rotatedPoints[i],omega);
    //}
    RigidMappingCuda3f_applyJ(points.size(), v, omega, out.deviceWrite(), rotatedPoints.deviceRead());
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
