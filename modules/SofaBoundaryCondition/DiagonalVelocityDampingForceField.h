/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>

namespace sofa
{

    namespace component
    {

        namespace forcefield
        {

            /// Apply damping forces to given degrees of freedom.
            template<class DataTypes>
            class DiagonalVelocityDampingForceField : public core::behavior::ForceField<DataTypes>
            {
            public:
                SOFA_CLASS(SOFA_TEMPLATE(DiagonalVelocityDampingForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

                typedef core::behavior::ForceField<DataTypes> Inherit;
                typedef typename DataTypes::VecCoord VecCoord;
                typedef typename DataTypes::VecDeriv VecDeriv;
                typedef typename DataTypes::Coord Coord;
                typedef typename DataTypes::Deriv Deriv;
                typedef typename Coord::value_type Real;
                typedef helper::vector<unsigned int> VecIndex;
                typedef core::objectmodel::Data<VecCoord> DataVecCoord;
                typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

                /// air drag coefficient.
                Data< VecDeriv > dampingCoefficients;

            protected:

                DiagonalVelocityDampingForceField();

            public:

                virtual void addForce (const core::MechanicalParams*, DataVecDeriv&, const DataVecCoord&, const DataVecDeriv&) override;

                virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df , const DataVecDeriv& d_dx) override;

                virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * /*m*/, SReal /*kFactor*/, unsigned int &/*offset*/) override {}

                virtual void addBToMatrix(sofa::defaulttype::BaseMatrix * mat, SReal bFact, unsigned int& offset) override;

                virtual SReal getPotentialEnergy(const core::MechanicalParams* params, const DataVecCoord& x) const override;

            };



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Vec3dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Vec2dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Vec1dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Vec6dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Rigid3dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Vec3fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Vec2fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Vec1fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Vec6fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Rigid3fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<defaulttype::Rigid2fTypes>;
#endif
#endif

        } // namespace forcefield

    } // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_H
