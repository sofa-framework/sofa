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
#ifndef FRAME_BLENDING_H
#define FRAME_BLENDING_H

#include "FrameMass.h"
#include "LinearBlending.h"
#include "DualQuatBlending.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <SofaBaseTopology/TopologyData.h>
#include <limits>

namespace sofa
{

namespace defaulttype
{

using sofa::defaulttype::FrameMass;


//      template<class In, class Out, class Material, int nbRef, int type>
//      class LinearBlending;

//      template<class In, class Out, class Material, int nbRef, int type>
//      class DualQuatBlending;



template< class Primitive>
class InDataTypesInfo
{
public:
    enum {VSize = Primitive::VSize};
};

template<int Dim, typename Real> class StdRigidTypes;

template< class Real, int Dim>
class InDataTypesInfo<StdRigidTypes<Dim,Real> >
{
public:
    enum {VSize = 6};
};



template< class Primitive>
class OutDataTypesInfo
{
public:
    typedef Vec<3,typename Primitive::Real> MaterialCoord;
    typedef vector<MaterialCoord> VecMaterialCoord;
    enum {primitive_order = 0}; ///< differential order: 0 for point, 1 for affine frame, 2 for quadratic frame
    enum {type = 0};            ///< used to tell apart rigid frames from affine frames (both are order 1)
};

template<int Spatial_dimensions, int Material_dimensions, int Order, typename Real> struct DeformationGradientTypes;

template< class Real, int Dim, int Order>
class OutDataTypesInfo<DeformationGradientTypes<Dim,Dim,Order,Real> >
{
public:
    typedef typename DeformationGradientTypes<Dim,Dim,Order,Real>::MaterialCoord MaterialCoord;
    typedef typename DeformationGradientTypes<Dim,Dim,Order,Real>::VecMaterialCoord VecMaterialCoord;
    enum {primitive_order = DeformationGradientTypes<Dim,Dim,Order,Real>::order};
    enum {type = DeformationGradientTypes<Dim,Dim,Order,Real>::order};
};


template< int N, typename real>
class OutDataTypesInfo<StdAffineTypes<N,real> >
{
public:
    typedef Vec<3,real> MaterialCoord;
    typedef vector<MaterialCoord > VecMaterialCoord;
    enum {primitive_order = 0};
    enum {type = 3};
};

template< int N, typename real>
class OutDataTypesInfo<StdRigidTypes<N,real> >
{
public:
    typedef Vec<3,real> MaterialCoord;
    typedef vector<MaterialCoord > VecMaterialCoord;
    enum {primitive_order = 0};
    enum {type = 4};
};

template< int N, typename real>
class OutDataTypesInfo<StdQuadraticTypes<N,real> >
{
public:
    typedef Vec<3,real> MaterialCoord;
    typedef vector<MaterialCoord > VecMaterialCoord;
    enum {primitive_order = 0};
    enum {type = 5};
};


template<bool IsPhysical>
class BaseFrameBlendingMapping : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BaseFrameBlendingMapping,IsPhysical),core::objectmodel::BaseObject);
    typedef Vec<3,double> Vec3d;

    static const bool isPhysical = IsPhysical;
    bool mappingHasChanged;
    vector<unsigned int> addedFrameIndices;
    vector<unsigned int> frameLife; // Test of removing frame
    Data<double> newFrameMinDist;
    Data<double> adaptativeCriteria;

    BaseFrameBlendingMapping ()
        : mappingHasChanged(false)
        , newFrameMinDist(initData ( &newFrameMinDist, std::numeric_limits<double>::max(), "newFrameMinDist","Minimal distance between inserted frames." ))
        , adaptativeCriteria(initData ( &adaptativeCriteria, 0.15, "adaptativeCriteria","Citeria to insert and remove frames." ))
    {
    }

    virtual bool insertFrame( const Vec3d& restPos) = 0;
    virtual void removeFrame( const unsigned int index) = 0;
};


template<class TIn, bool IsPhysical>
class FrameData : public virtual BaseFrameBlendingMapping<IsPhysical>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(FrameData,TIn,IsPhysical),SOFA_TEMPLATE(BaseFrameBlendingMapping,IsPhysical));
    // Input types
    typedef TIn In;
    typedef typename In::Real InReal;
    static const unsigned int num_spatial_dimensions=In::spatial_dimensions;
    enum {InVSize= defaulttype::InDataTypesInfo<In>::VSize};
    typedef FrameMass<num_spatial_dimensions,InVSize,InReal> FrameMassType;
    typedef sofa::component::topology::PointData<sofa::helper::vector<FrameMassType> > VecMass;
    typedef helper::vector<FrameMassType> MassVector;

    FrameData ()
        : BaseFrameBlendingMapping<IsPhysical> ()
    {
    }
    virtual void LumpMassesToFrames (MassVector& f_mass0, MassVector& f_mass) = 0;
};



template<class TOut, bool IsPhysical>
class SampleData : public virtual BaseFrameBlendingMapping<IsPhysical>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SampleData,TOut,IsPhysical),SOFA_TEMPLATE(BaseFrameBlendingMapping,IsPhysical));
    // Output types
    typedef TOut Out;
    typedef typename OutDataTypesInfo<Out>::MaterialCoord MaterialCoord;
    typedef typename OutDataTypesInfo<Out>::VecMaterialCoord VecMaterialCoord;

    Data<VecMaterialCoord> f_materialPoints;

    SampleData ()
        : BaseFrameBlendingMapping<IsPhysical> ()
        , f_materialPoints ( initData ( &f_materialPoints,"materialPoints","Coordinates of the samples in object space" ) )
    {
    }

    virtual void apply( MaterialCoord& coord, const MaterialCoord& restCoord) = 0; // Allow to tranfsorm a voxel from restPos for example
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dual Quaternion Blending
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace defaulttype
} // namespace sofa



#endif
