/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MATERIAL_HOOKEMATERIAL_INL
#define SOFA_COMPONENT_MATERIAL_HOOKEMATERIAL_INL

#include "NewHookeMaterial.h"


namespace sofa
{
namespace component
{
namespace material
{

//            template<class MaterialTypes>
//            HookeMaterial2<MaterialTypes>::HookeMaterial2()
//                : youngModulus(initData(&youngModulus, (Real)1.0, "youngModulus", "Stiffness, typically denoted using symbol E"))
//                , poissonRatio(initData(&poissonRatio, (Real)0.3, "poissonRatio", "Volume conservation, typically denoted using symbol \nu. Should be positive and less than 0.5 in 3d, respectively 1 in 2d. 0 means no volume conservation and 0.5 (resp. 1) means perfect volume conservation. Since a value of 0.5 (resp. 1) leads to a divison by 0, a smaller value should be used instead."))
//            {
//
//            }
//
//            template<class MaterialTypes>
//            void HookeMaterial2<MaterialTypes>::reinit()
//            {
//                Real f = youngModulus.getValue()/(1 - poissonRatio.getValue() * poissonRatio.getValue());
//                stressDiagonal = f;
//                stressOffDiagonal = poissonRatio.getValue() * f;
//                shear = f * (1 - poissonRatio.getValue()) /2;
//            }
//
//            // WARNING : The strain is defined as exx, eyy, 2exy
//            template<class MaterialTypes>
//            void HookeMaterial2<MaterialTypes>::computeStress  ( VecStr& stress, const VecStr& strain, const VecStr&, const VecMaterialCoord& )
//            {
//                for(unsigned int i=0; i<stress.size(); i++)
//                {
//                    stress[i][0] += stressDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1];
//                    stress[i][1] += stressOffDiagonal * strain[i][0] + stressDiagonal * strain[i][1];
//                    stress[i][2] += shear * strain[i][2];
//                }
//            }

template<class MaterialTypes>
HookeMaterial3<MaterialTypes>::HookeMaterial3()
    : bulkModulus(initData(&bulkModulus, (Real)1.0, "bulkModulus", "bulkModulus, to prevent from inversion of the deformation gradient."))
    , youngModulus(initData(&youngModulus, (Real)1.0, "youngModulus", "Stiffness, typically denoted using symbol E"))
    , poissonRatio(initData(&poissonRatio, (Real)0.0, "poissonRatio", "Volume conservation, typically denoted using symbol \nu. Should be positive and less than 0.5 in 3d, respectively 1 in 2d. 0 means no volume conservation and 0.5 (resp. 1) means perfect volume conservation. Since a value of 0.5 (resp. 1) leads to a divison by 0, a smaller value should be used instead."))
{
    //                reinit();
    //                Inherited::reinit();

}

template<class MaterialTypes>
void HookeMaterial3<MaterialTypes>::init()
{
    reinit();
    Inherited::init();
}

template<class MaterialTypes>
void HookeMaterial3<MaterialTypes>::reinit()
{
    Real f = youngModulus.getValue()/((1 + poissonRatio.getValue())*(1 - 2 * poissonRatio.getValue()));
    stressDiagonal = f * (1 - poissonRatio.getValue());
    stressOffDiagonal = poissonRatio.getValue() * f;
    shear = f * (1 - 2 * poissonRatio.getValue());// /2;
    Inherited::reinit();
}

template<class MaterialTypes>
void HookeMaterial3<MaterialTypes>::getStressStrainMatrix( StrStr& materialMatrix, const MaterialCoord& /*points*/ ) const
{
    Real young = this->youngModulus.getValue();
    Real poisson = this->poissonRatio.getValue();

    materialMatrix[0][0] = materialMatrix[1][1] = materialMatrix[2][2] = 1;
    materialMatrix[0][1] = materialMatrix[0][2] = materialMatrix[1][0] =
            materialMatrix[1][2] = materialMatrix[2][0] = materialMatrix[2][1] = poisson/(1-poisson);
    materialMatrix[0][3] = materialMatrix[0][4] = materialMatrix[0][5] = 0;
    materialMatrix[1][3] = materialMatrix[1][4] = materialMatrix[1][5] = 0;
    materialMatrix[2][3] = materialMatrix[2][4] = materialMatrix[2][5] = 0;
    materialMatrix[3][0] = materialMatrix[3][1] = materialMatrix[3][2] =
            materialMatrix[3][4] = materialMatrix[3][5] = 0;
    materialMatrix[4][0] = materialMatrix[4][1] = materialMatrix[4][2] =
            materialMatrix[4][3] = materialMatrix[4][5] = 0;
    materialMatrix[5][0] = materialMatrix[5][1] = materialMatrix[5][2] =
            materialMatrix[5][3] = materialMatrix[5][4] = 0;
    materialMatrix[3][3] = materialMatrix[4][4] = materialMatrix[5][5] =
            (1-2*poisson)/(2*(1-poisson));
    materialMatrix *= (young*(1-poisson))/((1+poisson)*(1-2*poisson));
}


// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void HookeMaterial3<MaterialTypes>::computeStress  ( VecStrain1& stresses, VecStrStr* stressStrainMatrices, const VecStrain1& strains, const VecStrain1& /*strainRates*/, const VecMaterialCoord& /*point*/  )
{
    for(unsigned int i=0; i<stresses.size(); i++)
    {
        stresses[i][0] = this->hookeStress(strains[i][0], stressDiagonal, stressOffDiagonal,  shear);

        if( stressStrainMatrices != NULL )
        {
            this->fillHookeMatrix( (*stressStrainMatrices)[i], stressDiagonal, stressOffDiagonal,  shear );
        }
    }
}


// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void HookeMaterial3<MaterialTypes>::computeStress  ( VecStrain4& stresses, VecStrStr* stressStrainMatrices, const VecStrain4& strains, const VecStrain4& /*strainRates*/, const VecMaterialCoord& /*point*/  )
{
    for(unsigned int i=0; i<stresses.size(); i++)
    {
        for(unsigned int j=0; j<4; j++)
        {
            stresses[i][j] = this->hookeStress(strains[i][j], stressDiagonal, stressOffDiagonal,  shear);
        }

        if( stressStrainMatrices != NULL )
        {
            this->fillHookeMatrix( (*stressStrainMatrices)[i], stressDiagonal, stressOffDiagonal,  shear );
        }
    }
}

// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void HookeMaterial3<MaterialTypes>::computeStress  ( VecStrain10& stresses, VecStrStr* stressStrainMatrices, const VecStrain10& strains, const VecStrain10& /*strainRates*/, const VecMaterialCoord& /*point*/  )
{
    for(unsigned int i=0; i<stresses.size(); i++)
    {
        for(unsigned int j=0; j<10; j++)
        {
            stresses[i][j] = this->hookeStress(strains[i][j], stressDiagonal, stressOffDiagonal,  shear);
        }

        if( stressStrainMatrices != NULL )
        {
            this->fillHookeMatrix( (*stressStrainMatrices)[i], stressDiagonal, stressOffDiagonal,  shear );
        }
    }
}

// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void HookeMaterial3<MaterialTypes>::computeStressChange  ( VecStrain1& stressChanges, const VecStrain1& strainChanges, const VecMaterialCoord& /*point*/  )
{
    for(unsigned int i=0; i<stressChanges.size(); i++)
    {
        stressChanges[i][0] = this->hookeStress(strainChanges[i][0], stressDiagonal, stressOffDiagonal,  shear);
    }
}
// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void HookeMaterial3<MaterialTypes>::computeStressChange  ( VecStrain4& stressChanges, const VecStrain4& strainChanges, const VecMaterialCoord& /*point*/  )
{
    for(unsigned int i=0; i<stressChanges.size(); i++)
    {
        for(unsigned int j=0; j<4; j++)
        {
            stressChanges[i][j] = this->hookeStress(strainChanges[i][j], stressDiagonal, stressOffDiagonal,  shear);
        }
    }
}
// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void HookeMaterial3<MaterialTypes>::computeStressChange  ( VecStrain10& stressChanges, const VecStrain10& strainChanges, const VecMaterialCoord& /*point*/  )
{
    for(unsigned int i=0; i<stressChanges.size(); i++)
    {
        for(unsigned int j=0; j<10; j++)
        {
            stressChanges[i][j] = this->hookeStress(strainChanges[i][j], stressDiagonal, stressOffDiagonal,  shear);
        }
    }
}

template<class MaterialTypes>
typename HookeMaterial3<MaterialTypes>::Real HookeMaterial3<MaterialTypes>::getBulkModulus(const unsigned int /*sampleindex*/) const
{
    return bulkModulus.getValue();
}

template<class MaterialTypes>
bool HookeMaterial3<MaterialTypes>::computeVolumeIntegrationFactors(const unsigned int /*sampleindex*/,const MaterialCoord& /*point*/,const unsigned int order,vector<Real>& moments)
{
    unsigned int dim=(order+1)*(order+2)*(order+3)/6; // complete basis in 3D
    moments.resize(dim);
    Real vol=1; // default volume of a gauss point
    Real dl=(Real)pow(vol,(Real)1./(Real)3.); // default width a the cube
    moments[0] = vol;
    if(order<2) return true;
    moments[4] = vol*dl*dl/(Real)12.;  moments[7] = vol*dl*dl/(Real)12.;  moments[9] = vol*dl*dl/(Real)12.;
    if(order<4) return true;
    moments[20] = vol*dl*dl*dl*dl/(Real)80.;  moments[21] = vol*dl*dl*dl*dl/(Real)144.;  moments[22] = vol*dl*dl*dl*dl/(Real)144.;  moments[23] = vol*dl*dl*dl*dl/(Real)80.;  moments[24] = vol*dl*dl*dl*dl/(Real)144.;  moments[25] = vol*dl*dl*dl*dl/(Real)80.;
    return true;
}

//// WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
//template<class MaterialTypes>
//void HookeMaterial3<MaterialTypes>::computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& )
//{
//    for(unsigned int i=0; i<stress.size(); i++)
//    {
//        stress[i][0] = stressDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//        stress[i][1] = stressOffDiagonal * strain[i][0] + stressDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//        stress[i][2] = stressOffDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressDiagonal * strain[i][2];
//        stress[i][3] = shear * strain[i][3];
//        stress[i][4] = shear * strain[i][4];
//        stress[i][5] = shear * strain[i][5];
//    }
//    if( stressStrainMatrices != NULL ){
//        VecStrStr&  m = *stressStrainMatrices;
//        m.resize( stress.size() );
//        m[0].fill(0);
//        m[0][0][0] = m[0][1][1] = m[0][2][2] = stressDiagonal;
//        m[0][0][1] = m[0][0][2] = m[0][1][0] = m[0][1][2] = m[0][2][0] = m[0][2][1] = stressOffDiagonal;
//        m[0][3][3] = m[0][4][4] = m[0][5][5] = shear;
//        for( unsigned int i=1; i<m.size(); i++ ){
//            m[i] = m[0];
//        }
//    }
//}

//// WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
//template<class MaterialTypes>
//void HookeMaterial3<MaterialTypes>::computeStress  ( VecElStr& stress, VecStrStr* stressStrainMatrices, const VecElStr& strain, const VecElStr& )
//{
//    for(unsigned int e=0; e<10; e++)
//    for(unsigned int i=0; i<stress.size(); i++)
//    {
//        stress[i][0][e] = stressDiagonal * strain[i][0][e] + stressOffDiagonal * strain[i][1][e] + stressOffDiagonal * strain[i][2][e];
//        stress[i][1][e] = stressOffDiagonal * strain[i][0][e] + stressDiagonal * strain[i][1][e] + stressOffDiagonal * strain[i][2][e];
//        stress[i][2][e] = stressOffDiagonal * strain[i][0][e] + stressOffDiagonal * strain[i][1][e] + stressDiagonal * strain[i][2][e];
//        stress[i][3][e] = shear * strain[i][3][e];
//        stress[i][4][e] = shear * strain[i][4][e];
//        stress[i][5][e] = shear * strain[i][5][e];
//    }
//    if( stressStrainMatrices != NULL ){
//        VecStrStr&  m = *stressStrainMatrices;
//        m.resize( stress.size() );
//        m[0].fill(0);
//        m[0][0][0] = m[0][1][1] = m[0][2][2] = stressDiagonal;
//        m[0][0][1] = m[0][0][2] = m[0][1][0] = m[0][1][2] = m[0][2][0] = m[0][2][1] = stressOffDiagonal;
//        m[0][3][3] = m[0][4][4] = m[0][5][5] = shear;
//        for( unsigned int i=1; i<m.size(); i++ ){
//            m[i] = m[0];
//        }
//    }
//}

//            // WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
//            template<class MaterialTypes>
//            void HookeMaterial3<MaterialTypes>::computeDStress  ( VecStr& stress, const VecStr& strain )
//            {
//                for(unsigned int i=0; i<stress.size(); i++)
//                {
//                    stress[i][0] = stressDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//                    stress[i][1] = stressOffDiagonal * strain[i][0] + stressDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//                    stress[i][2] = stressOffDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressDiagonal * strain[i][2];
//                    stress[i][3] = shear * strain[i][3];
//                    stress[i][4] = shear * strain[i][4];
//                    stress[i][5] = shear * strain[i][5];
//                }
//            }




}

} // namespace component

} // namespace sofa

#endif


