#ifndef SOFA_GPU_CUDA_CUDAPOINTMODEL_H
#define SOFA_GPU_CUDA_CUDAPOINTMODEL_H

#include "CudaTypes.h"

#include <sofa/core/CollisionModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/component/topology/MeshTopology.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

using namespace sofa::defaulttype;
//using namespace sofa::component::collision;

class CudaPointModel;

class CudaPoint : public core::TCollisionElementIterator<CudaPointModel>
{
public:
    CudaPoint(CudaPointModel* model, int index);

    explicit CudaPoint(core::CollisionElementIterator& i);
};

class CudaPointModel : public core::CollisionModel, public core::VisualModel
{
public:
    typedef CudaVec3fTypes InDataTypes;
    typedef CudaVec3fTypes DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef CudaPoint Element;
    friend class CudaPoint;

    DataField<int> groupSize;

    CudaPointModel();

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    //virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(int index);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }


    core::componentmodel::behavior::MechanicalState<InDataTypes>* getMechanicalState() { return mstate; }

    //virtual const char* getTypeName() const { return "Point"; }

protected:

    core::componentmodel::behavior::MechanicalState<InDataTypes>* mstate;
};

inline CudaPoint::CudaPoint(CudaPointModel* model, int index)
    : core::TCollisionElementIterator<CudaPointModel>(model, index)
{}

inline CudaPoint::CudaPoint(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<CudaPointModel>(static_cast<CudaPointModel*>(i.getCollisionModel()), i.getIndex())
{
}

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
