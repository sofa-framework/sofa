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
#ifndef SOFA_COMPONENT_MAPPING_HexahedronCompositeFEMMapping_H
#define SOFA_COMPONENT_MAPPING_HexahedronCompositeFEMMapping_H

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

    virtual void init();

    virtual void apply( const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecCoord& out, const InDataVecCoord& in);
    //void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    virtual void applyJ( const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecDeriv& out, const InDataVecDeriv& in);
    //void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    virtual void applyJT( const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, InDataVecDeriv& out, const OutDataVecDeriv& in);
    //void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    virtual void applyJT( const sofa::core::ConstraintParams* cparams /* PARAMS FIRST */, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in)
    {
        serr << "applyJT(constraint) not implemented" << sendl;
    }

    //void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );

    void draw(const core::visual::VisualParams* vparams);

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

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
