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
#ifndef SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_INL

#include "DiagonalVelocityDampingForceField.h"


namespace sofa
{

    namespace component
    {

        namespace forcefield
        {


            template<class DataTypes>
            DiagonalVelocityDampingForceField<DataTypes>::DiagonalVelocityDampingForceField()
                : dampingCoefficients(initData(&dampingCoefficients, VecDeriv(1), "dampingCoefficient", "velocity damping coefficient"))
            {
            }


            template<class DataTypes>
            void DiagonalVelocityDampingForceField<DataTypes>::init()
            {
                Inherit::init();
            }

            template<class DataTypes>
            void DiagonalVelocityDampingForceField<DataTypes>::addForce(const core::MechanicalParams* /*params*/ /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& p, const DataVecDeriv& v)
            {
                sofa::helper::WriteAccessor<DataVecDeriv> force(f);
                const VecDeriv& velocity = v.getValue();
                unsigned nbDampingCoeff = dampingCoefficients.getValue().size();
                if(nbDampingCoeff>0)
                {
                    for(unsigned i=0; i<velocity.size();i++)
                        for(unsigned j=0; j<Deriv::total_size; j++)
                            if(nbDampingCoeff>=i)
                                force[i][j] -= velocity[i][j]*dampingCoefficients.getValue()[i][j];
                            else
                                force[i][j] -= velocity[i][j]*dampingCoefficients.getValue()[nbDampingCoeff-1][j];
                }
            }

            template<class DataTypes>
            void DiagonalVelocityDampingForceField<DataTypes>::addBToMatrix(sofa::defaulttype::BaseMatrix * mat, double bFact, unsigned int& offset)
            {
                unsigned nbDampingCoeff = dampingCoefficients.getValue().size();
                for(unsigned i=0; i<mat->rowSize()/Deriv::total_size; i++)
                    for(unsigned j=0; j<Deriv::total_size; j++)
                        if(nbDampingCoeff<i)
                            mat->add(i*Deriv::total_size+j,i*Deriv::total_size+j,-dampingCoefficients.getValue()[i][j]*bFact);
                        else
                            mat->add(i*Deriv::total_size+j,i*Deriv::total_size+j,-dampingCoefficients.getValue()[nbDampingCoeff-1][j]*bFact);
            }

            template <class DataTypes>
            double DiagonalVelocityDampingForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*params*/ /* PARAMS FIRST */, const DataVecCoord& x) const
            {
                return 0;
            }


            template<class DataTypes>
            void DiagonalVelocityDampingForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
            {
                //                sofa::helper::WriteAccessor<Inherit::getMState()> force(f);
            }

        } // namespace forcefield

    } // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_AIRDRAGFORCEFIELD_INL



