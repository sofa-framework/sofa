/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/mapping/linear/config.h>

#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

#include <vector>

#include <memory>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
class BeamLinearMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BeamLinearMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef Out OutDataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;

    typedef In InDataTypes;
    typedef typename In::Deriv InDeriv;

    typedef typename Coord::value_type Real;
    enum { N    = OutDataTypes::spatial_dimensions               };
    enum { NIn  = sofa::defaulttype::DataTypeInfo<InDeriv>::Size };
    enum { NOut = sofa::defaulttype::DataTypeInfo<Deriv>::Size   };
    typedef type::Mat<N, N, Real> Mat;
    typedef type::Vec<N, Real> Vector;
    typedef type::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<MBloc> MatrixType;

protected:
    type::vector<Coord> points;
    //Coord translation;
    //Real orientation[4];
    //Mat rotation;
    sofa::type::vector<Real> beamLength;
    sofa::type::vector<Coord> rotatedPoints0;
    sofa::type::vector<Coord> rotatedPoints1;

    std::unique_ptr<MatrixType> matrixJ;
    bool updateJ;

    BeamLinearMapping()
        : Inherit()
        //, index(initData(&index,(unsigned)0,"index","input DOF index"))
        , matrixJ()
        , updateJ(false)
        , localCoord(initData(&localCoord,true,"localCoord","true if initial coordinates are in the beam local coordinate system (i.e. a point at (10,0,0) is on the DOF number 10, whereas if this is false it is at whatever position on the beam where the distance from the initial DOF is 10)"))
    {
    }

    virtual ~BeamLinearMapping()
    {
    }

public:
    //Data<unsigned> index;
    Data<bool> localCoord; ///< true if initial coordinates are in the beam local coordinate system (i.e. a point at (10,0,0) is on the DOF number 10, whereas if this is false it is at whatever position on the beam where the distance from the initial DOF is 10)

    void init() override;

    void apply(const core::MechanicalParams *mparams, Data< typename Out::VecCoord >& out, const Data< typename In::VecCoord >& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data< typename Out::VecDeriv >& out, const Data< typename In::VecDeriv >& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data< typename In::VecDeriv >& out, const Data< typename Out::VecDeriv >& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in) override;

    const sofa::linearalgebra::BaseMatrix* getJ() override;

    void draw(const core::visual::VisualParams* vparams) override;
};



template <std::size_t N, class Real> struct RigidMappingMatrixHelper;


#if !defined(SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API BeamLinearMapping< defaulttype::Rigid3Types, defaulttype::Vec3Types >;
#endif

} // namespace sofa::component::mapping::linear
