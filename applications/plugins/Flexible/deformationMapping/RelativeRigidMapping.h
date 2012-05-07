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
#ifndef SOFA_COMPONENT_MAPPING_RelativeRigidMapping_H
#define SOFA_COMPONENT_MAPPING_RelativeRigidMapping_H

#include <sofa/core/Mapping.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/component/component.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

#include <sofa/defaulttype/RigidTypes.h>


#include "../initFlexible.h"

#include "../utils/se3.h"


namespace sofa
{
using helper::vector;

namespace component
{

namespace mapping
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class RelativeRigidMappingInternalData
{
public:
};


/**
    Maps a collection of rigids to the relative space:
    (a, b, c) -> (inv(a) * b, inb(b) * c)

@author Maxime Tournier
  */
template <class TIn, class TOut>
class SOFA_Flexible_API RelativeRigidMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(RelativeRigidMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef linearsolver::EigenSparseMatrix<TIn,TOut>    SparseMatrixEigen;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };
    typedef defaulttype::Mat<Out::deriv_total_size, In::deriv_total_size,Real>  Block;

    virtual void init();

    virtual void apply(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<OutVecCoord>& out, const Data<InVecCoord>& in);

    virtual void applyJ(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in);

    virtual void applyJT(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in);

    virtual void applyJT(const core::ConstraintParams *cparams /* PARAMS FIRST */, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in);

//    virtual void applyDJT(const core::MechanicalParams* mparams /* PARAMS FIRST  = core::MechanicalParams::defaultInstance()*/, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce );

    virtual const sofa::defaulttype::BaseMatrix* getJ();
    virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs();

    virtual void draw(const core::visual::VisualParams* vparams);

protected:
    RelativeRigidMapping();
    virtual ~RelativeRigidMapping();

    SparseMatrixEigen jacobian;                           ///< Jacobian of the mapping
    vector< defaulttype::BaseMatrix* > baseMatrices;      ///< Jacobian of the mapping, in a vector

    typedef unsigned int index_type;
    typedef vector< defaulttype::Vec<2, index_type> > edges_type;

    // stores mapped pairs (p, c) -> inv(p) * c
    Data< edges_type > edges;

    typedef SE3<Real> se3_type;

    typedef typename se3_type::mat66 mat66;
    typedef typename se3_type::coord_type coord_type;

    // mapping block: d( (a, b) -> inv(a) * b )
    static void blocks(mat66& Ja, mat66& Jb,
            const coord_type& a, const coord_type& b)
    {

        se3_type se3;

        coord_type diff = se3.prod(se3.inv(a), b);

        Ja = -se3.ad( se3.inv(diff) );
        Jb.setIdentity();

        // TODO optimize body/sofa conversions
        Ja = se3.sofa(diff) * Ja * se3.body(a);
        Jb = se3.sofa(diff) * Jb * se3.body(b);
    }

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_RelativeRigidMapping_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_RIGID_API RelativeRigidMapping< Rigid3dTypes, Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_RIGID_API RelativeRigidMapping< Rigid3fTypes, Rigid3fTypes >;
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
