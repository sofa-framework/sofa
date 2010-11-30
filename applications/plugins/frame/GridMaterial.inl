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
#ifndef SOFA_COMPONENT_MATERIAL_HOOKEMATERIAL_INL
#define SOFA_COMPONENT_MATERIAL_HOOKEMATERIAL_INL

#include "GridMaterial.h"


namespace sofa
{
namespace component
{
namespace material
{


template<class MaterialTypes>
GridMaterial<MaterialTypes>::GridMaterial()
{
}

template<class MaterialTypes>
void GridMaterial<MaterialTypes>::init()
{

    vector<VoxelGridLoader*> vg;
    sofa::core::objectmodel::BaseContext* context=  this->getContext();
    context->get<VoxelGridLoader>( &vg, core::objectmodel::BaseContext::Local);
    assert(vg.size()>0);
    this->voxelGridLoader = vg[0];

    Inherited::init();
}

// WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
template<class MaterialTypes>
void GridMaterial<MaterialTypes>::computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& )
{

//                Real f = youngModulus.getValue()/((1 + poissonRatio.getValue())*(1 - 2 * poissonRatio.getValue()));
//                stressDiagonal = f * (1 - poissonRatio.getValue());
//                stressOffDiagonal = poissonRatio.getValue() * f;
//                shear = f * (1 - 2 * poissonRatio.getValue()) /2;
//
//
//                for(unsigned i=0; i<stress.size(); i++)
//                {
//                    stress[i][0] = stressDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//                    stress[i][1] = stressOffDiagonal * strain[i][0] + stressDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//                    stress[i][2] = stressOffDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressDiagonal * strain[i][2];
//                    stress[i][3] = shear * strain[i][3];
//                    stress[i][4] = shear * strain[i][4];
//                    stress[i][5] = shear * strain[i][5];
//                }
//                if( stressStrainMatrices != NULL ){
//                    VecStrStr&  m = *stressStrainMatrices;
//                    m.resize( stress.size() );
//                    m[0].fill(0);
//                    m[0][0][0] = m[0][1][1] = m[0][2][2] = stressDiagonal;
//                    m[0][0][1] = m[0][0][2] = m[0][1][0] = m[0][1][2] = m[0][2][0] = m[0][2][1] = stressOffDiagonal;
//                    m[0][3][3] = m[0][4][4] = m[0][5][5] = shear;
//                    for( unsigned i=1; i<m.size(); i++ ){
//                        m[i] = m[0];
//                    }
//                }
}

// WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
template<class MaterialTypes>
void GridMaterial<MaterialTypes>::computeStress  ( VecElStr& stress, VecStrStr* stressStrainMatrices, const VecElStr& strain, const VecElStr& )
{
//                Real f = youngModulus.getValue()/((1 + poissonRatio.getValue())*(1 - 2 * poissonRatio.getValue()));
//                stressDiagonal = f * (1 - poissonRatio.getValue());
//                stressOffDiagonal = poissonRatio.getValue() * f;
//                shear = f * (1 - 2 * poissonRatio.getValue()) /2;
//
//
//                for(unsigned e=0; e<10; e++)
//                for(unsigned i=0; i<stress.size(); i++)
//                {
//                    stress[i][0][e] = stressDiagonal * strain[i][0][e] + stressOffDiagonal * strain[i][1][e] + stressOffDiagonal * strain[i][2][e];
//                    stress[i][1][e] = stressOffDiagonal * strain[i][0][e] + stressDiagonal * strain[i][1][e] + stressOffDiagonal * strain[i][2][e];
//                    stress[i][2][e] = stressOffDiagonal * strain[i][0][e] + stressOffDiagonal * strain[i][1][e] + stressDiagonal * strain[i][2][e];
//                    stress[i][3][e] = shear * strain[i][3][e];
//                    stress[i][4][e] = shear * strain[i][4][e];
//                    stress[i][5][e] = shear * strain[i][5][e];
//                }
//                if( stressStrainMatrices != NULL ){
//                    VecStrStr&  m = *stressStrainMatrices;
//                    m.resize( stress.size() );
//                    m[0].fill(0);
//                    m[0][0][0] = m[0][1][1] = m[0][2][2] = stressDiagonal;
//                    m[0][0][1] = m[0][0][2] = m[0][1][0] = m[0][1][2] = m[0][2][0] = m[0][2][1] = stressOffDiagonal;
//                    m[0][3][3] = m[0][4][4] = m[0][5][5] = shear;
//                    for( unsigned i=1; i<m.size(); i++ ){
//                        m[i] = m[0];
//                    }
//                }
}





}

} // namespace component

} // namespace sofa

#endif


