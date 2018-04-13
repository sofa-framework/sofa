/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDAPENALITYCONTACTFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDAPENALITYCONTACTFORCEFIELD_H

#include "CudaTypes.h"
#include <SofaObjectInteraction/PenalityContactForceField.h>
#include <sofa/gpu/cuda/CudaCollisionDetection.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

} // namespace cuda

} // namespace gpu

namespace component
{

namespace interactionforcefield
{

using sofa::gpu::cuda::CudaVec3fTypes;

template<>
class PenalityContactForceField<CudaVec3fTypes> : public core::behavior::PairInteractionForceField<CudaVec3fTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PenalityContactForceField,CudaVec3fTypes),SOFA_TEMPLATE(core::behavior::PairInteractionForceField,CudaVec3fTypes));
    typedef CudaVec3fTypes DataTypes;
    typedef core::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef DataTypes DataTypes1;
    typedef DataTypes DataTypes2;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Coord::value_type Real;
    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

//protected:
    /*
    struct Contact
    {
        int m1, m2;   ///< the two extremities of the spring: masses m1 and m2
        Deriv norm;   ///< contact normal, from m1 to m2
        Real dist;    ///< minimum distance between the points
        Real ks;      ///< spring stiffness
        Real mu_s;    ///< coulomb friction coefficient (currently unused)
        Real mu_v;    ///< viscous friction coefficient
        Real pen;     ///< current penetration
        int age;      ///< how old is this contact
    };
    */
    sofa::gpu::cuda::CudaVector<sofa::defaulttype::Vec4f> contacts;
    sofa::gpu::cuda::CudaVector<float> pen;

    // contacts from previous frame
    //std::vector<Contact> prevContacts;

//public:

    PenalityContactForceField(MechanicalState* object1, MechanicalState* object2)
        : Inherit(object1, object2)
    {
    }

    PenalityContactForceField()
    {
    }

    void clear(int reserve = 0);

    void addContact(int m1, int m2, const Deriv& norm, Real dist, Real ks, Real mu_s = 0.0f, Real mu_v = 0.0f, int oldIndex = 0);

    void setContacts(Real distance, Real ks, sofa::core::collision::GPUDetectionOutputVector* outputs, bool useDistance, defaulttype::Mat3x3f* normXForm = NULL);

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f1, DataVecDeriv& d_f2, const DataVecCoord& d_x1, const DataVecCoord& d_x2, const DataVecDeriv& d_v1, const DataVecDeriv& d_v2) override;

    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df1, DataVecDeriv& d_df2, const DataVecDeriv& d_dx1, const DataVecDeriv& d_dx2) override;

    virtual SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x1, const DataVecCoord& x2) const override;

    void draw(const core::visual::VisualParams*) override;
};

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
