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
#ifndef SOFA_COMPONENT_MATERIAL_HOOKEMATERIAL_H
#define SOFA_COMPONENT_MATERIAL_HOOKEMATERIAL_H

#include "initFrame.h"
#include "NewMaterial.h"


namespace sofa
{
namespace component
{
namespace material
{

using namespace sofa::defaulttype;
//template<class TMaterialTypes, int NumMaterialCoordinates> class HookeMaterial;
///** \brief Material implementing Hooke's law.
// */
//template<class TMaterialTypes, int NumMaterialCoordinates>
//class SOFA_COMPONENT_FEM_API HookeMaterial : public Material<TMaterialTypes>
//{
//public:
//    typedef Material<TMaterialTypes> Inherited;
////    SOFA_CLASS( SOFA_TEMPLATE(HookeMaterial, TMaterialTypes, NumMaterialCoordinates), SOFA_TEMPLATE(Inherited, TMaterialTypes) );
//
//    typedef TMaterialTypes MaterialTypes;
//    typedef typename MaterialTypes::Real Real;        ///< Scalar values.
//    typedef typename MaterialTypes::MaterialCoord MaterialCoord;        ///< Material coordinates of a point in the object.
//    typedef typename MaterialTypes::VecMaterialCoord VecMaterialCoord;  ///< Vector of material coordinates.
//    typedef typename MaterialTypes::Str Str;            ///< Strain or stress tensor defined as a vector with 6 entries for 3d material coordinates, 3 entries for 2d coordinates, and 1 entry for 1d coordinates.
//    typedef typename MaterialTypes::VecStr VecStr;      ///< Vector of strain or stress tensors
//
//    HookeMaterial();
//    virtual ~HookeMaterial(){}
//
//    /// Recompute the stress-strain matrix when the parameters are changed.
//    virtual void reinit();
//
//    /// implementation of the abstract function
//    virtual void computeStress  ( VecStr& stress, const VecStr& strain, const VecStr& strainRate, const VecMaterialCoord& point );
//    /// implementation of the abstract function
//    virtual void computeDStress ( VecStr& stressChange, const VecStr& strainChange, const VecMaterialCoord& point );
//
//    Data<Real> youngModulus;  ///< Stiffness, typically denoted using symbol \f$ E \f$
//
//    /** \brief Volume conservation, typically denoted using symbol \f$  \nu \f$.
//    Should be positive and less than 0.5 in 3d, respectively 1 in 2d.
//    0 means no volume conservation, while 0.5 (resp. 1) means perfect volume conservation.
//    Since a value of 0.5 (resp. 1) leads to a divison by 0, a smaller value should be used instead.
//    */
//    Data<Real> poissonRatio;
//
//protected:
//    Real stressDiagonal, stressOffDiagonal, shear;
//};

//template<class TMaterialTypes>
//class SOFA_COMPONENT_FEM_API HookeMaterial2 : public Material<TMaterialTypes>
//{
//public:
//    typedef Material<TMaterialTypes> Inherited;
////    SOFA_CLASS( SOFA_TEMPLATE(HookeMaterial, TMaterialTypes, NumMaterialCoordinates), SOFA_TEMPLATE(Inherited, TMaterialTypes) );
//
//    typedef TMaterialTypes MaterialTypes;
//    typedef typename MaterialTypes::Real Real;        ///< Scalar values.
//    typedef typename MaterialTypes::MaterialCoord MaterialCoord;        ///< Material coordinates of a point in the object.
//    typedef typename MaterialTypes::VecMaterialCoord VecMaterialCoord;  ///< Vector of material coordinates.
//    typedef typename MaterialTypes::Str Str;            ///< Strain or stress tensor defined as a vector with 6 entries for 3d material coordinates, 3 entries for 2d coordinates, and 1 entry for 1d coordinates.
//    typedef typename MaterialTypes::VecStr VecStr;      ///< Vector of strain or stress tensors
//
//    HookeMaterial2();
//    virtual ~HookeMaterial2(){}
//
//    /// Recompute the stress-strain matrix when the parameters are changed.
//    virtual void reinit();
//
//    /// implementation of the abstract function
//    virtual void computeStress  ( VecStr& stress, const VecStr& strain, const VecStr& strainRate, const VecMaterialCoord& point );
//    /// implementation of the abstract function
//    virtual void computeDStress ( VecStr& stressChange, const VecStr& strainChange, const VecMaterialCoord& point );
//
//    Data<Real> youngModulus;  ///< Stiffness, typically denoted using symbol \f$ E \f$
//
//    /** \brief Volume conservation, typically denoted using symbol \f$  \nu \f$.
//    Should be positive and less than 0.5 in 3d, respectively 1 in 2d.
//    0 means no volume conservation, while 0.5 (resp. 1) means perfect volume conservation.
//    Since a value of 0.5 (resp. 1) leads to a divison by 0, a smaller value should be used instead.
//    */
//    Data<Real> poissonRatio;
//
//protected:
//    Real stressDiagonal, stressOffDiagonal, shear;
//};

template<class TMaterialTypes>
class SOFA_FRAME_API HookeMaterial3 : public Material<TMaterialTypes>
{
public:
    typedef Material<TMaterialTypes> Inherited;
    SOFA_CLASS( SOFA_TEMPLATE(HookeMaterial3, TMaterialTypes), SOFA_TEMPLATE(Material, TMaterialTypes) );

    typedef typename Inherited::Real Real;        ///< Scalar values.
    typedef typename Inherited::Str Str;            ///< Strain or stress tensor defined as a vector with 6 entries for 3d material coordinates, 3 entries for 2d coordinates, and 1 entry for 1d coordinates.
    typedef typename Inherited::Strain1 Strain1;
    typedef typename Inherited::VecStrain1 VecStrain1;
    typedef typename Inherited::Strain4 Strain4;
    typedef typename Inherited::VecStrain4 VecStrain4;
    typedef typename Inherited::Strain10 Strain10;
    typedef typename Inherited::VecStrain10 VecStrain10;

//    typedef typename Inherited::VecStr VecStr;      ///< Vector of strain or stress tensors
    typedef typename Inherited::StrStr StrStr;      ///< Stress-strain matrix
    typedef typename Inherited::VecStrStr VecStrStr;      ///< Vector of Stress-strain matrices
    typedef typename Inherited::VecMaterialCoord VecMaterialCoord;
    typedef typename Inherited::MaterialCoord MaterialCoord;

    HookeMaterial3();
    virtual ~HookeMaterial3() {}

    /// Compute the stress-strain matrix
    virtual void init();

    /// Recompute the stress-strain matrix when the parameters are changed.
    virtual void reinit();

    //typedef defaulttype::DeformationGradient<3,3,1,Real> DeformationGradient331;
    //typedef typename DeformationGradient331::SampleIntegVector SampleIntegVector331;
    //typedef vector<SampleIntegVector331>  VecSampleIntegVector331;
    //typedef typename DeformationGradient331::Strain            Strain331;
    //typedef vector<Strain331>  VecStrain331;

    ///** \brief Compute stress based on local strain and strain rate at each point.
    //*/
    //virtual void computeStress  ( VecStrain331& stress, const VecStrain331& strain, const VecStrain331& strainRate, const VecSampleIntegVector331& integ ){}

    ///** \brief Compute stress change based on strain change
    // */
    //virtual void computeStressChange  ( VecStrain331& stressChange, const VecStrain331& strainChange, const VecSampleIntegVector331& integ ){}


    //typedef defaulttype::DeformationGradient<3,3,2,Real> DeformationGradient332;
    //typedef typename DeformationGradient332::SampleIntegVector SampleIntegVector332;
    //typedef vector<SampleIntegVector332>  VecSampleIntegVector332;
    //typedef typename DeformationGradient332::Strain            Strain332;
    //typedef vector<Strain332>  VecStrain332;

    ///** \brief Compute stress based on local strain and strain rate at each point.
    //*/
    //virtual void computeStress  ( VecStrain332& stress, const VecStrain332& strain, const VecStrain332& strainRate, const VecSampleIntegVector332& integ ){}

    ///** \brief Compute stress change based on strain change
    // */
    //virtual void computeStressChange  ( VecStrain332& stressChange, const VecStrain332& strainChange, const VecSampleIntegVector332& integ ){}


    /// implementation of the abstract function
    virtual Real getBulkModulus(const unsigned int sampleindex) const;
    virtual bool computeVolumeIntegrationFactors(const unsigned int sampleindex,const MaterialCoord& point,const unsigned int order,vector<Real>& moments);
    virtual void computeStress  ( VecStrain1& stress, VecStrStr* stressStrainMatrices, const VecStrain1& strain, const VecStrain1& strainRate, const VecMaterialCoord& point );
    virtual void computeStress  ( VecStrain4& stress, VecStrStr* stressStrainMatrices, const VecStrain4& strain, const VecStrain4& strainRate, const VecMaterialCoord& point );
    virtual void computeStress  ( VecStrain10& stress, VecStrStr* stressStrainMatrices, const VecStrain10& strain, const VecStrain10& strainRate, const VecMaterialCoord& point );
    virtual void computeStressChange  ( VecStrain1& stressChange, const VecStrain1& strainChange, const VecMaterialCoord& point );
    virtual void computeStressChange  ( VecStrain4& stressChange, const VecStrain4& strainChange, const VecMaterialCoord& point );
    virtual void computeStressChange  ( VecStrain10& stressChange, const VecStrain10& strainChange, const VecMaterialCoord& point );
    //virtual void computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& strainRate );
    //virtual void computeStress  ( VecElStr& stress, VecStrStr* stressStrainMatrices, const VecElStr& strain, const VecElStr& strainRate );

    /// get the StressStrain matrices at the given points, assuming null strain or linear material
    virtual void getStressStrainMatrix( StrStr& matrix, const MaterialCoord& point ) const;


    Data<Real> bulkModulus;  ///< bulkModulus, to prevent from inversion of the deformation gradient

    Data<Real> youngModulus;  ///< Stiffness, typically denoted using symbol \f$ E \f$

    /** \brief Volume conservation, typically denoted using symbol \f$  \nu \f$.
    Should be positive and less than 0.5 in 3d, respectively 1 in 2d.
    0 means no volume conservation, while 0.5 (resp. 1) means perfect volume conservation.
    Since a value of 0.5 (resp. 1) leads to a divison by 0, a smaller value should be used instead.
    */
    Data<Real> poissonRatio;

    static const char* Name();

    std::string getTemplateName() const
    {
        return templateName(this);
    }
    static std::string templateName(const HookeMaterial3<TMaterialTypes>* = NULL)
    {
        return TMaterialTypes::Name();
    }

protected:
    Real stressDiagonal, stressOffDiagonal, shear; // entries of the stress-strain matrix
};

#ifdef SOFA_FLOAT
template<> inline const char* HookeMaterial3<Material3d>::Name() { return "HookeMaterial3d"; }
template<> inline const char* HookeMaterial3<Material3f>::Name() { return "HookeMaterial3"; }
#else
template<> inline const char* HookeMaterial3<Material3d>::Name() { return "HookeMaterial3"; }
template<> inline const char* HookeMaterial3<Material3f>::Name() { return "HookeMaterial3f"; }
#endif


}

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FEM_BASEMATERIAL_H
