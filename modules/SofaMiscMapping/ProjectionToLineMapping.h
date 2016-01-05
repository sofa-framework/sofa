/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_DistanceToLineMapping_H
#define SOFA_COMPONENT_MAPPING_DistanceToLineMapping_H
#include "config.h"

#include <sofa/core/Mapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
//#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{



/** Maps point positions to their projections on a fixed target line.
    Only a subset of the parent points is mapped. This can be used to constrain the trajectories of one or several particles.

    In: parent point positions

    Out: orthogonal projection of each point on a target line

    @author Matthieu Nesme
  */
template <class TIn, class TOut>
class ProjectionToTargetLineMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(ProjectionToTargetLineMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

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
    typedef defaulttype::Vec<In::spatial_dimensions> Direction;

    Data< vector<unsigned> > f_indices;         ///< indices of the parent points
    Data< OutVecCoord >      f_origins; ///< origins of the lines the point is projected to
    Data< OutVecCoord >      f_directions; ///< directions of the lines the point is projected to (should be normalized, and are normalized in init)
    Data< SReal >            d_drawScale; ///< drawing scale
    Data< defaulttype::Vec4f >  d_drawColor; ///< drawing color

    virtual void init();
    virtual void reinit();

    virtual void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in);

    virtual void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in);

    virtual void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in);

    virtual void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in);


    virtual const sofa::defaulttype::BaseMatrix* getJ();
    virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs();


    virtual void draw(const core::visual::VisualParams* vparams);


    // no geometric stiffness
    virtual void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*parentForce*/, core::ConstMultiVecDerivId /*childForce*/ ){}
    virtual void updateK(const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId /*childForce*/ ){}
    virtual const defaulttype::BaseMatrix* getK(){ return NULL; }

    virtual void updateForceMask();


protected:
    ProjectionToTargetLineMapping();
    virtual ~ProjectionToTargetLineMapping() {}

    SparseMatrixEigen jacobian;                      ///< Jacobian of the mapping
    vector<defaulttype::BaseMatrix*> baseMatrices;   ///< Jacobian of the mapping, in a vector
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_ProjectionToLineMapping_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API ProjectionToTargetLineMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_MISC_MAPPING_API ProjectionToTargetLineMapping< defaulttype::Rigid3dTypes, defaulttype::Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API ProjectionToTargetLineMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API ProjectionToTargetLineMapping< defaulttype::Rigid3fTypes, defaulttype::Vec3fTypes >;
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
