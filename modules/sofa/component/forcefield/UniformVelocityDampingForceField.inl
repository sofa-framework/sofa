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
#ifndef SOFA_COMPONENT_FORCEFIELD_UNIFORMVELOCITYDAMPINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_UNIFORMVELOCITYDAMPINGFORCEFIELD_INL

#include "UniformVelocityDampingForceField.h"

namespace sofa
{

    namespace component
    {

        namespace forcefield
        {


            template<class DataTypes>
            UniformVelocityDampingForceField<DataTypes>::UniformVelocityDampingForceField()
                : dampingCoefficient(initData(&dampingCoefficient, 0.1, "dampingCoefficient", "velocity damping coefficient"))
            {
            }


            template<class DataTypes>
            void UniformVelocityDampingForceField<DataTypes>::init()
            {
                Inherit::init();
            }

            template<class DataTypes>
            void UniformVelocityDampingForceField<DataTypes>::addForce(const core::MechanicalParams* /*params*/ /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& p, const DataVecDeriv& v)
            {
                sofa::helper::WriteAccessor<DataVecDeriv> force(f);
                const VecDeriv& velocity = v.getValue();

                 for(unsigned int i=0; i<velocity.size(); i++)
                     force[i] -= velocity[i]*dampingCoefficient.getValue();
            }

            template<class DataTypes>
            void UniformVelocityDampingForceField<DataTypes>::addBToMatrix(sofa::defaulttype::BaseMatrix * mat, double bFact, unsigned int& offset)
            {
                for(unsigned i=0; i<mat->rowSize(); i++)
                    mat->add(i,i,-dampingCoefficient.getValue()*bFact);
            }

            template <class DataTypes>
            double UniformVelocityDampingForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*params*/ /* PARAMS FIRST */, const DataVecCoord& x) const
            {
               return 0;
            }


            template<class DataTypes>
            void UniformVelocityDampingForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
            {
//                sofa::helper::WriteAccessor<Inherit::getMState()> force(f);
            }

        } // namespace forcefield

    } // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_AIRDRAGFORCEFIELD_INL



