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
#ifndef SOFA_COMPONENT_MAPPING_SquareDistanceMapping_H
#define SOFA_COMPONENT_MAPPING_SquareDistanceMapping_H
#include "config.h"

#include <sofa/core/Mapping.h>
#include <sofa/core/MultiMapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/RGBAColor.h>


namespace sofa
{

namespace component
{

namespace mapping
{



/** Maps point positions to square distances.
  Type TOut corresponds to a scalar value.
  The pairs are given in an EdgeSetTopologyContainer in the same node.

    In: parent point positions
    Out: square distance between point pairs, minus a square rest distance.

    No restLength (imposed null rest length) for now
    TODO: compute Jacobians for non null restLength

@author Matthieu Nesme
  */


// If the rest lengths are not defined, they are set using the initial values.
// If computeDistance is set to true, the rest lengths are set to 0.
template <class TIn, class TOut>
class SquareDistanceMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SquareDistanceMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

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
    typedef linearsolver::EigenSparseMatrix<TIn,TOut>   SparseMatrixEigen;
    typedef linearsolver::EigenSparseMatrix<TIn,TIn>    SparseKMatrixEigen;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };
    typedef topology::EdgeSetTopologyContainer::SeqEdges SeqEdges;
    typedef defaulttype::Vec<In::spatial_dimensions,Real> Direction;


//    Data< bool >		   f_computeDistance;	///< computeDistance = true ---> restDistance = 0
//    Data< helper::vector< Real > > f_restLengths;		///< rest length of each link
    Data< Real >           d_showObjectScale;   ///< drawing size
    Data< defaulttype::RGBAColor > d_color;         ///< drawing color
    Data< unsigned >       d_geometricStiffness; ///< how to compute geometric stiffness (0->no GS, 1->exact GS, 2->stabilized GS)

    virtual void init() override;

    using Inherit::apply;

    virtual void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in) override;

    virtual void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in) override;

    virtual void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in) override;

    virtual void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in) override;

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce ) override;

    virtual const sofa::defaulttype::BaseMatrix* getJ() override;
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs() override;

    virtual void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForce ) override;
    virtual const defaulttype::BaseMatrix* getK() override;

    virtual void draw(const core::visual::VisualParams* vparams) override;

    virtual void updateForceMask() override;

protected:
    SquareDistanceMapping();
    virtual ~SquareDistanceMapping();

    topology::EdgeSetTopologyContainer* edgeContainer;  ///< where the edges are defined
    SparseMatrixEigen jacobian;                         ///< Jacobian of the mapping
    helper::vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector
    SparseKMatrixEigen K;                               ///< Assembled geometric stiffness matrix

    /// r=b-a only for position (eventual rotation, affine transform... remains null)
    void computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b );
};




#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_SquareDistanceMapping_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API SquareDistanceMapping< defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_MISC_MAPPING_API SquareDistanceMapping< defaulttype::Rigid3dTypes, defaulttype::Vec1dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API SquareDistanceMapping< defaulttype::Vec3fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_MISC_MAPPING_API SquareDistanceMapping< defaulttype::Rigid3fTypes, defaulttype::Vec1fTypes >;
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
