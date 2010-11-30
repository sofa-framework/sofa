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
    typedef typename Inherited::VecStr VecStr;      ///< Vector of strain or stress tensors
    typedef typename Inherited::El2Str ElStr;            ///< Elaston strain or stress, see DefaultMaterialTypes
    typedef typename Inherited::VecEl2Str VecElStr;      ///< Vector of elaston strain or stress
    typedef typename Inherited::StrStr StrStr;      ///< Stress-strain matrix
    typedef typename Inherited::VecStrStr VecStrStr;      ///< Vector of Stress-strain matrices

    HookeMaterial3();
    virtual ~HookeMaterial3() {}

    /// Compute the stress-strain matrix
    virtual void init();

    /// Recompute the stress-strain matrix when the parameters are changed.
    virtual void reinit();

    /// implementation of the abstract function
    virtual void computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& strainRate );

    virtual void computeStress  ( VecElStr& stress, VecStrStr* stressStrainMatrices, const VecElStr& strain, const VecElStr& strainRate );

//    /// implementation of the abstract function
//    virtual void computeDStress ( VecStr& stressChange, const VecStr& strainChange );

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

//#ifdef SOFA_FLOAT
//template<> inline const char* HookeMaterial3<Material3d>::Name() { return "HookeMaterial3d"; }
//template<> inline const char* HookeMaterial3<Material3f>::Name() { return "HookeMaterial3"; }
//#else
//template<> inline const char* HookeMaterial3<Material3d>::Name() { return "HookeMaterial3"; }
//template<> inline const char* HookeMaterial3<Material3f>::Name() { return "HookeMaterial3f"; }
//#endif


}

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FEM_BASEMATERIAL_H
