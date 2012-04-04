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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_GreenStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_GreenStrainMAPPING_H

#include "initFlexible.h"
#include <sofa/core/Mapping.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

#include "GreenStrainJacobianBlock.h"

#include <sofa/component/linearsolver/EigenSparseMatrix.h>


namespace sofa
{


namespace component
{

namespace mapping
{

using helper::vector;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class GreenStrainMappingInternalData
{
public:
};


/** Deformation Gradient to Green Lagrangian Strain mapping
*/

template <class TIn, class TOut>
class SOFA_Flexible_API GreenStrainMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(GreenStrainMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;

    /** @name  Input types    */
    //@{
    typedef TIn In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real Real;
    //@}

    /** @name  Output types    */
    //@{
    typedef TOut Out;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    enum { spatial_dimensions = Out::spatial_dimensions };
    //@}

    /** @name  Jacobian types    */
    //@{
//    typedef typename Block::MatBlock  MatBlock;
    //@}

    /** @name  Jacobian types    */
    //@{
    typedef defaulttype::GreenStrainJacobianBlock<In,Out>  Block;  ///< Jacobian block object
    typedef vector<Block>  SparseMatrix;

    typedef typename Block::MatBlock  MatBlock;  ///< Jacobian block matrix
    typedef linearsolver::EigenSparseMatrix<In,Out>    SparseMatrixEigen;
    //@}


    /** @name Mapping functions */
    //@{
    virtual void init();
    virtual void reinit();
    virtual void apply(const core::MechanicalParams *mparams , Data<OutVecCoord>& out, const Data<InVecCoord>& in);
    virtual void applyJ(const core::MechanicalParams *mparams , Data<OutVecDeriv>& out, const Data<InVecDeriv>& in);
    virtual void applyJT(const core::MechanicalParams *mparams , Data<InVecDeriv>& out, const Data<OutVecDeriv>& in);
    virtual void applyJT(const core::ConstraintParams *cparams , Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in);
    //    virtual void applyDJT(const core::MechanicalParams* mparams /* PARAMS FIRST  = core::MechanicalParams::defaultInstance()*/, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce );
    const defaulttype::BaseMatrix* getJ(const core::MechanicalParams *mparams);
    void draw(const core::visual::VisualParams* vparams);
    //@}

protected:
    GreenStrainMapping (core::State<In>* from = NULL, core::State<Out>* to= NULL);
    virtual ~GreenStrainMapping();

    Data<bool> assembleJ;
    void updateJ();

    SparseMatrix jacobian;   ///< Jacobian of the mapping
    SparseMatrixEigen eigenJacobian;  ///< Assembled Jacobian matrix

    helper::ParticleMask* maskFrom;  ///< Subset of master DOF, to cull out computations involving null forces or displacements
    helper::ParticleMask* maskTo;    ///< Subset of slave DOF, to cull out computations involving null forces or displacements
};


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
