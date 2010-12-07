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
#ifndef SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_H
#define SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_H

#include "initFrame.h"
#include <sofa/component/mapping/SkinningMapping.h>

#include "NewMaterial.h"
#include "GridMaterial.h"


namespace sofa
{

namespace component
{

namespace mapping
{

using helper::vector;
using sofa::component::material::MaterialTypes;
using sofa::component::material::GridMaterial;

/** This class is not supposed to be instanciated. It is only used to contain what is common between its derived classes.
  */
template <class TIn, class TOut>
class FrameBlendingMapping : public SkinningMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(FrameBlendingMapping,TIn,TOut), SOFA_TEMPLATE2(SkinningMapping,TIn,TOut) );

    typedef SkinningMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef Out DataTypes;

    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    //typedef typename Out::SpatialCoord SpatialCoord;


    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;

    typedef MaterialTypes<3,Real> materialType;
    typedef typename sofa::component::material::GridMaterial<materialType>::Vec3  materialCoordType;
    typedef typename sofa::component::material::GridMaterial<materialType>::VecVec3  materialVecCoordType;

    GridMaterial< materialType>* gridMaterial;
    Data<unsigned int> targetFrameNumber;
    Data<unsigned int> targetSampleNumber;

public:
    FrameBlendingMapping (core::State<In>* from, core::State<Out>* to );
    virtual ~FrameBlendingMapping();

    void init();

    virtual void apply(typename Out::VecCoord& out, const typename In::VecCoord& in);
    virtual void applyJ(typename Out::VecDeriv& out, const typename In::VecDeriv& in);
    virtual void applyJT(typename In::VecDeriv& out, const typename Out::VecDeriv& in);
    virtual void applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in);
};


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
