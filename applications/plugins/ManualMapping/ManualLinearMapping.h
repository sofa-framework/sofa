/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_MAPPING_MANUALLINEARMAPPING_H
#define SOFA_COMPONENT_MAPPING_MANUALLINEARMAPPING_H

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vector>
#include <memory>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <ManualMapping/config.h>


namespace sofa
{

namespace component
{

namespace mapping
{


/// A purely linear interpolation (where xc=J.xp and no geometric stiffness)
/// Note that Coord and Deriv are necessarly of the same type
template <class TIn, class TOut>
class ManualLinearMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(ManualLinearMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename In::Real			Real;
    typedef typename In::VecCoord		InVecCoord;
    typedef typename In::VecDeriv		InVecDeriv;
    typedef typename In::Coord			InCoord;
    typedef typename In::Deriv			InDeriv;
    typedef typename In::MatrixDeriv	InMatrixDeriv;

    typedef typename Out::VecCoord		VecCoord;
    typedef typename Out::VecDeriv		VecDeriv;
    typedef typename Out::Coord			Coord;
    typedef typename Out::Deriv			Deriv;
    typedef typename Out::MatrixDeriv	MatrixDeriv;

    typedef Out OutDataTypes;
    typedef typename OutDataTypes::Real     OutReal;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;

    enum
    {
        NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size
    };
    enum
    {
        NOut = sofa::defaulttype::DataTypeInfo<Deriv>::Size
    };


    typedef linearsolver::EigenSparseMatrix<TIn, TOut> eigen_type;
    eigen_type eigen;

    typedef helper::vector< defaulttype::BaseMatrix* > js_type;
    js_type js;


    /// This matrix defines the mapping and must be manually given
    eigen_type _matrixJ;


protected:

    virtual ~ManualLinearMapping()
    {
    }

    void updateJ();

public:

    void init();

    void apply(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<VecCoord>& out, const Data<InVecCoord>& in);

    void applyJ(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<VecDeriv>& out, const Data<InVecDeriv>& in);

    void applyJT(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<InVecDeriv>& out, const Data<VecDeriv>& in);

    void applyJT(const core::ConstraintParams *cparams /* PARAMS FIRST */, Data<InMatrixDeriv>& out, const Data<MatrixDeriv>& in);

    const sofa::defaulttype::BaseMatrix* getJ();
    const js_type* getJs();

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_MANUALLINEARMAPPING_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_ManualMapping_API ManualLinearMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_ManualMapping_API ManualLinearMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_ManualMapping_API ManualLinearMapping< defaulttype::Vec3dTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_ManualMapping_API ManualLinearMapping< defaulttype::Vec3fTypes, defaulttype::Vec3dTypes >;
#endif
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
