/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDASPHEREMODEL_H
#define SOFA_GPU_CUDA_CUDASPHEREMODEL_H

#include "CudaTypes.h"

#include <sofa/core/CollisionModel.h>
#include <sofa/component/container/MechanicalObject.h>
#include "CudaMechanicalObject.h"

namespace sofa
{

namespace gpu
{

namespace cuda
{

using namespace sofa::defaulttype;
/* typedef sofa::component::collision::SphereModel <gpu::cuda::CudaVec3fTypes> CudaSphereModel; */

class CudaSphereModel;
/* typedef sofa::component::collision::Sphere <gpu::cuda::CudaVec3fTypes> CudaSphere; */

class CudaSphere : public core::TCollisionElementIterator<CudaSphereModel>
{
public:
    typedef SReal Real;
    typedef CudaVec3fTypes::Coord Coord;

    CudaSphere(CudaSphereModel* model, int index);

    explicit CudaSphere(const core::CollisionElementIterator& i);


    const Coord& center() const;
    const Coord& p() const;
    const Coord& pFree() const;
    const Coord& v() const;
    SReal r() const;
};

class CudaSphereModel : public core::CollisionModel
{
public:
    SOFA_CLASS(CudaSphereModel,core::CollisionModel);
    typedef CudaVec3fTypes InDataTypes;
    typedef CudaVec3fTypes DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef DataTypes::Real Real;
    typedef DataTypes::VecReal VecReal;
    typedef CudaSphere Element;
    friend class CudaSphere;

    CudaSphereModel();

    virtual void init();

    virtual bool load(const char* filename);

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    //virtual void computeContinuousBoundingTree(double dt, int
    //maxDepth=0);

    void draw(int index);

    void draw();

    virtual void drawColourPicking(const ColourCode method);
    virtual sofa::defaulttype::Vector3 getPositionFromWeights(int index, Real /*a*/ ,Real /*b*/, Real /*c*/);

    core::behavior::MechanicalState<InDataTypes>* getMechanicalState() { return mstate; }

    const VecReal& getR() const { return this->radius.getValue(); }

    void setRadius(const int i, const Real r);
    void setRadius(const Real r);

    int addSphere(const Vector3& pos, Real r);
    void setSphere(int i, const Vector3& pos, Real r);


protected:
    core::behavior::MechanicalState<InDataTypes>* mstate;

    sofa::core::objectmodel::DataFileName filename;

    Data< VecReal > radius;
    Data< SReal > defaultRadius;

    Real getRadius(const int i) const; /// return the radius of the given sphere

    class Loader;


};

inline CudaSphere::CudaSphere(CudaSphereModel* model, int index)
    : core::TCollisionElementIterator<CudaSphereModel>(model, index)
{}

inline CudaSphere::CudaSphere(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<CudaSphereModel>(static_cast<CudaSphereModel*>(i.getCollisionModel()), i.getIndex())
{}


inline const CudaSphere::Coord& CudaSphere::center() const { return (*model->mstate->getX())[index]; }

inline const CudaSphere::Coord& CudaSphere::p() const { return (*model->mstate->getX())[index]; }

inline const CudaSphere::Coord& CudaSphere::pFree() const { return (*model->mstate->getXfree())[index]; }

inline const CudaSphere::Coord& CudaSphere::v() const { return (*model->mstate->getV())[index]; }

inline CudaSphere::Real CudaSphere::r() const { return (CudaSphere::Real) model->getRadius((unsigned)index); }

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
