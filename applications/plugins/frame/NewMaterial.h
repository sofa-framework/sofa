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
#ifndef SOFA_COMPONENT_MATERIAL_MATERIAL_H
#define SOFA_COMPONENT_MATERIAL_MATERIAL_H

#include <sofa/component/component.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include "initFrame.h"


namespace sofa
{
namespace component
{
namespace material
{

using namespace sofa::defaulttype;

/** \brief Material as a stress-strain relationship.
 */
template<class TMaterialTypes>
class SOFA_COMPONENT_FEM_API Material : public virtual core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(Material, TMaterialTypes), Inherited);

    typedef TMaterialTypes MaterialTypes;
    typedef typename MaterialTypes::Str Str;            ///< Strain or stress tensor defined as a vector with 6 entries for 3d material coordinates, 3 entries for 2d coordinates, and 1 entry for 1d coordinates.
    typedef typename MaterialTypes::VecStr VecStr;      ///< Vector of strain or stress tensors
    typedef typename MaterialTypes::StrStr StrStr;      ///< Stress-strain matrix
    typedef typename MaterialTypes::VecStrStr VecStrStr;      ///< Vector of Stress-strain matrices

    virtual ~Material() {}

    /** \brief Compute stress based on local strain and strain rate at each point.
      The stress-strain relation may depend on strain rate (time derivative of strain).
    */
    virtual void computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& strainRate ) = 0;

//    /** \brief Compute stress change based on local strain.
//      This is for using in implicit methods.
//    */
//    virtual void computeDStress ( VecStr& stressChange, const VecStr& strainChange ) = 0;

};

template<int D, class R>
struct DefaultMaterialTypes
{
    typedef R Real;
    static const int N = D*(D+1)/2;             ///< Number of independent entries in the symmetric DxD strain tensor

    typedef defaulttype::Vec<N,R> Str;       ///< Strain or stress tensor in Voigt (i.e. vector) notation
    typedef helper::vector<Str> VecStr;

    typedef defaulttype::Mat<N,N,R> StrStr;  ///< Stress-strain matrix
    typedef helper::vector<StrStr> VecStrStr;

    static const char* Name();
};

typedef DefaultMaterialTypes<3,float> Material3f;
typedef DefaultMaterialTypes<3,double> Material3d;

#ifdef SOFA_FLOAT
template<> inline const char* Material3d::Name() { return "Material3d"; }
template<> inline const char* Material3f::Name() { return "Material"; }
#else
template<> inline const char* Material3d::Name() { return "Material"; }
template<> inline const char* Material3f::Name() { return "Material3f"; }
#endif



} //

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FEM_BASEMATERIAL_H
