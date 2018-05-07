/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_HexahedronCompositeFEMMapping_H
#define SOFA_COMPONENT_MAPPING_HexahedronCompositeFEMMapping_H
#include "config.h"

//#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/core/Mapping.h>
#include <sofa/core/State.h>
#include <SofaBaseTopology/SparseGridTopology.h>
#include <SofaNonUniformFem/HexahedronCompositeFEMForceFieldAndMass.h>

#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Mat.h>

namespace sofa
{

namespace component
{


namespace mapping
{

template <class BasicMapping>
class HexahedronCompositeFEMMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedronCompositeFEMMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In            In;
    typedef sofa::core::State<In>*  ptrStateIn;
    typedef typename Inherit::Out           Out;
    typedef sofa::core::State<Out>* ptrStateOut;
    //typedef typename Out::DataTypes OutDataTypes;
    typedef Out OutDataTypes;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    //typedef typename In::DataTypes InDataTypes;
    typedef In InDataTypes;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename OutCoord::value_type Real;


    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

    typedef topology::SparseGridTopology SparseGridTopologyT;
    typedef sofa::component::forcefield::HexahedronCompositeFEMForceFieldAndMass<In> HexahedronCompositeFEMForceFieldAndMassT;


    typedef defaulttype::Mat<3,8*3> Weight;
    typedef typename HexahedronCompositeFEMForceFieldAndMassT::Transformation Transformation;
    typedef helper::fixed_array< InCoord, 8 > Nodes;

protected:
    HexahedronCompositeFEMMapping (  ): Inherit ( )
    {
// 		_method = initData(&this->_method,0,"method","0: auto, 1: coarseNodes->surface, 2: coarseNodes->finestNodes->surface");
        _alreadyInit=false;
    }
public:
    virtual ~HexahedronCompositeFEMMapping() {}

    virtual void init() override;

    virtual void apply( const sofa::core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in) override;
    //void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    virtual void applyJ( const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in) override;
    //void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    virtual void applyJT( const sofa::core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in) override;
    //void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    virtual void applyJT( const sofa::core::ConstraintParams* /*cparams*/, InDataMatrixDeriv& /*out*/, const OutDataMatrixDeriv& /*in*/) override
    {
        serr << "applyJT(constraint) not implemented" << sendl;
    }

    //void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );

    void draw(const core::visual::VisualParams* vparams) override;

// 	Data<int> _method;

protected :

    bool _alreadyInit;


    helper::vector< OutCoord > _finePos;

    // in order to treat large dispacements in translation (rotation is given by the corotational force field)
// 	  InVecCoord _baycenters0;
// 	  InCoord computeTranslation( const SparseGridTopologyT::Hexa& hexa, unsigned idx );
    OutVecCoord _p0; // intial position of the interpolated vertices
    InVecCoord _qCoarse0, _qFine0; // intial position of the element nodes
    InVecCoord _qFine; // only for drawing

// 	  helper::vector< helper::Quater<Real> > _rotations;
    helper::vector< Transformation >  _rotations;


// 	  helper::vector< helper::vector<unsigned > > _pointsCorrespondingToElem; // in which element is the interpolated vertex?
    helper::vector< Weight > _weights; // a weight matrix for each vertex, such as dp=W.dq with q the 8 values of the embedding element

    // for method 2
    helper::vector< std::pair< int, helper::fixed_array<Real,8> > > _finestBarycentricCoord; // barycentric coordinates for each mapped points into the finest elements (fine element idx + weights)

    helper::vector< std::map< int, Weight > > _finestWeights; // for each fine nodes -> a list of incident coarse element idx and the corresponding weight

    // necessary objects
    SparseGridTopologyT* _sparseGrid;
    SparseGridTopologyT::SPtr _finestSparseGrid;
    HexahedronCompositeFEMForceFieldAndMassT* _forcefield;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_HEXAHEDRONCOMPOSITEFEMMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class HexahedronCompositeFEMMapping< core::Mapping< defaulttype::Vec3dTypes, defaulttype::ExtVec3fTypes > >;
extern template class HexahedronCompositeFEMMapping< core::Mapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes > >;
#endif
#ifndef SOFA_DOUBLE
extern template class HexahedronCompositeFEMMapping< core::Mapping< defaulttype::Vec3fTypes, defaulttype::ExtVec3fTypes > >;
extern template class HexahedronCompositeFEMMapping< core::Mapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes > >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class HexahedronCompositeFEMMapping< core::Mapping< defaulttype::Vec3fTypes, defaulttype::Vec3dTypes > >;
extern template class HexahedronCompositeFEMMapping< core::Mapping< defaulttype::Vec3dTypes, defaulttype::Vec3fTypes > >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
