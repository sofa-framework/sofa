/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_H

#include <sofa/core/behavior/ForceField.h>
#include <sofa/component/component.h>

namespace sofa
{

    namespace component
    {

        namespace forcefield
        {

            /// Apply air drag forces to given degrees of freedom.
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


            public:
                /// air drag coefficient.
                Data< VecDeriv > dampingCoefficients;

            protected:
                DiagonalVelocityDampingForceField();
            public:
                /// Init function
                void init();

                /// Add the forces
                virtual void addForce (const core::MechanicalParams* params /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

                virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df , const DataVecDeriv& d_dx){}

                virtual void addKToMatrix(sofa::defaulttype::BaseMatrix *m, SReal kFactor, unsigned int &offset){}

                virtual void addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/, double /*kFact*/){}

                virtual void addBToMatrix(sofa::defaulttype::BaseMatrix * mat, SReal bFact, unsigned int& offset);

                virtual double getPotentialEnergy(const core::MechanicalParams* params /* PARAMS FIRST */, const DataVecCoord& x) const;

                void draw(const core::visual::VisualParams* vparams);

            };

#ifndef SOFA_FLOAT
            using sofa::defaulttype::Vec1dTypes;
            using sofa::defaulttype::Vec2dTypes;
            using sofa::defaulttype::Vec3dTypes;
            using sofa::defaulttype::Vec6dTypes;
            using sofa::defaulttype::Rigid2dTypes;
            using sofa::defaulttype::Rigid3dTypes;
#endif

#ifndef SOFA_DOUBLE
            using sofa::defaulttype::Vec1fTypes;
            using sofa::defaulttype::Vec2fTypes;
            using sofa::defaulttype::Vec3fTypes;
            using sofa::defaulttype::Vec6fTypes;
            using sofa::defaulttype::Rigid2fTypes;
            using sofa::defaulttype::Rigid3fTypes;
#endif


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec3dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec2dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec1dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec6dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Rigid3dTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec3fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec2fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec1fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec6fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Rigid3fTypes>;
            extern template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Rigid2fTypes>;
#endif
#endif

        } // namespace forcefield

    } // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_H
