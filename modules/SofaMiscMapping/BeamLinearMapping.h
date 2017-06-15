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
#ifndef SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_H
#define SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_H
#include "config.h"

#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

#include <vector>

#include <memory>


namespace sofa
{

namespace component
{

namespace mapping
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
    typedef defaulttype::Mat<N, N, Real> Mat;
    typedef defaulttype::Vec<N, Real> Vector;
    typedef defaulttype::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<MBloc> MatrixType;

protected:
    helper::vector<Coord> points;
    //Coord translation;
    //Real orientation[4];
    //Mat rotation;
    sofa::helper::vector<Real> beamLength;
    sofa::helper::vector<Coord> rotatedPoints0;
    sofa::helper::vector<Coord> rotatedPoints1;

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
    Data<bool> localCoord;

    void init();

    void apply(const core::MechanicalParams *mparams, Data< typename Out::VecCoord >& out, const Data< typename In::VecCoord >& in);

    void applyJ(const core::MechanicalParams *mparams, Data< typename Out::VecDeriv >& out, const Data< typename In::VecDeriv >& in);

    void applyJT(const core::MechanicalParams *mparams, Data< typename In::VecDeriv >& out, const Data< typename Out::VecDeriv >& in);

    void applyJT(const core::ConstraintParams *cparams, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in);

    const sofa::defaulttype::BaseMatrix* getJ();

    void draw(const core::visual::VisualParams* vparams);
};



template <int N, class Real> struct RigidMappingMatrixHelper;


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API BeamLinearMapping< defaulttype::Rigid3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_MISC_MAPPING_API BeamLinearMapping< defaulttype::Rigid3dTypes, defaulttype::ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API BeamLinearMapping< defaulttype::Rigid3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API BeamLinearMapping< defaulttype::Rigid3fTypes, defaulttype::ExtVec3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API BeamLinearMapping< defaulttype::Rigid3dTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API BeamLinearMapping< defaulttype::Rigid3fTypes, defaulttype::Vec3dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif

