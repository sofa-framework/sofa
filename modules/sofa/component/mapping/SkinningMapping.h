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
#ifndef SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H
#define SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H

#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MappedModel.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/helper/SVector.h>

#include <vector>

#include <sofa/component/mapping/BasicSkinningMapping.h>
#include <sofa/component/component.h>
#include <sofa/helper/OptionsGroup.h>

#ifdef SOFA_DEV
#include <sofa/component/topology/HexahedronGeodesicalDistance.h>
#endif

namespace sofa
{

namespace component
{

namespace mapping
{

using sofa::helper::vector;
using sofa::helper::Quater;
using sofa::helper::SVector;
#ifdef SOFA_DEV
using sofa::component::topology::HexahedronGeodesicalDistance;
#endif








template <class BasicMapping>
class SkinningMapping : public BasicSkinningMapping<BasicMapping>
{
public:
    SOFA_CLASS ( SOFA_TEMPLATE ( SkinningMapping,BasicMapping ), SOFA_TEMPLATE ( BasicSkinningMapping, BasicMapping ) );
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::DataTypes DataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename defaulttype::SparseConstraint<Deriv> OutSparseConstraint;
    typedef typename OutSparseConstraint::const_data_iterator OutConstraintIterator;

    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::Real InReal;
    typedef typename Out::Real Real;
    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    //typedef defaulttype::Mat<3,1,Real> Mat31;
    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef defaulttype::Mat<3,6,Real> Mat36;
    typedef vector<Mat36> VMat36;
    typedef vector<VMat36> VVMat36;
    typedef defaulttype::Mat<3,8,Real> Mat38;
    typedef defaulttype::Mat<4,4,Real> Mat44;
    //typedef defaulttype::Mat<6,1,Real> Mat61;
    typedef defaulttype::Mat<6,3,Real> Mat63;
    typedef defaulttype::Mat<6,6,Real> Mat66;
    typedef vector<Mat66> VMat66;
    typedef vector<VMat66> VVMat66;
    //typedef defaulttype::Mat<8,1,Real> Mat81;
    typedef defaulttype::Mat<8,3,Real> Mat83;
    typedef defaulttype::Mat<8,6,Real> Mat86;
    typedef vector<Mat86> VMat86;
    typedef defaulttype::Mat<8,8,Real> Mat88;
    typedef vector<Mat88> VMat88;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef vector<Vec3> VVec3;
    typedef vector<VVec3> VVVec3;
    typedef defaulttype::Vec<4,Real> Vec4;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef vector<Vec6> VVec6;
    typedef vector<VVec6> VVVec6;
    typedef defaulttype::Vec<8,Real> Vec8;
    typedef Quater<InReal> Quat;
    typedef sofa::helper::vector< VecCoord > VecVecCoord;
    typedef SVector<double> VD;
    typedef SVector<SVector<double> > VVD;

#ifdef SOFA_DEV
    // These typedef are here to avoid compilation pb encountered with ResizableExtVect Type.
    typedef typename sofa::defaulttype::StdVectorTypes<sofa::defaulttype::Vec<N, double>, sofa::defaulttype::Vec<N, double>, double> GeoType; // = Vec3fTypes or Vec3dTypes
    typedef typename HexahedronGeodesicalDistance< GeoType >::VecCoord GeoVecCoord;
    typedef typename HexahedronGeodesicalDistance< GeoType >::Coord GeoCoord;
    typedef typename HexahedronGeodesicalDistance< GeoType >::VecVecCoord GeoVecVecCoord;
#else
    typedef Coord GeoCoord;
    typedef VecCoord GeoVecCoord;
#endif

public:
    SkinningMapping ( In* from, Out* to );
    virtual ~SkinningMapping();

};

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif
#endif




} // namespace mapping

} // namespace component

} // namespace sofa

#endif
