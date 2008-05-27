#ifndef OGLTETRAHEDRALMODEL_H_
#define OGLTETRAHEDRALMODEL_H_

#include <sofa/core/VisualModel.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec3Types.h>

#ifdef SOFA_GPU_CUDA
#include <sofa/gpu/cuda/CudaTypes.h>
#endif

namespace sofa
{
namespace component
{
namespace visualmodel
{

template<class DataTypes>
class OglTetrahedralModel : public core::VisualModel
{
public:
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
#ifdef SOFA_GPU_CUDA
    typedef sofa::gpu::cuda::CudaVec3fTypes CudaVec3fTypes;
#endif

private:
    core::componentmodel::topology::BaseMeshTopology* topo;
    core::componentmodel::behavior::MechanicalState<DataTypes>* nodes;

    Data<bool> depthTest;
    Data<bool> blending;

public:
    OglTetrahedralModel();
    virtual ~OglTetrahedralModel();

    void init();
    void drawVisual();
    bool addBBox(double* minBBox, double* maxBBox);

};

#endif /*OGLTETRAHEDRALMODEL_H_*/
}
}
}
