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
#ifndef SOFA_COMPONENT_MAPPING_DistanceFromTargetMapping_H
#define SOFA_COMPONENT_MAPPING_DistanceFromTargetMapping_H
#include "config.h"

#include <sofa/core/Mapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/RGBAColor.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class DistanceFromTargetMappingInternalData
{
public:
};


struct BaseDistanceFromTargetMapping
{
    virtual void updateTarget( unsigned index, SReal x, SReal y, SReal z ) = 0;
};



/** Maps point positions to distances from target points.
    Only a subset of the parent points is mapped. This can be used to constrain the trajectories of one or several particles.

    In: parent point positions

    Out: distance from each point to a target position, minus a rest distance.

    (changed class name on Feb. 4, 2014, previous name was DistanceMapping)



  @author Francois Faure
  */
template <class TIn, class TOut>
class DistanceFromTargetMapping : public core::Mapping<TIn, TOut>, public BaseDistanceFromTargetMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DistanceFromTargetMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

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
    typedef linearsolver::EigenSparseMatrix<In,In>    SparseKMatrixEigen;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };
    typedef defaulttype::Vec<In::spatial_dimensions> Direction;
    typedef typename Inherit::ForceMask ForceMask;

    Data< helper::vector<unsigned> > f_indices;         ///< indices of the parent points
    Data< InVecCoord >       f_targetPositions; ///< positions the distances are measured from
    Data< helper::vector< Real > >   f_restDistances;   ///< rest distance from each position
    Data< unsigned >         d_geometricStiffness; ///< how to compute geometric stiffness (0->no GS, 1->exact GS, 2->stabilized GS)

    /// Add a target with a desired distance
    void createTarget( unsigned index, const InCoord& position, Real distance);

    /// Update the position of a target
    void updateTarget( unsigned index, const InCoord& position);
    virtual void updateTarget( unsigned index, SReal x, SReal y, SReal z ) override;

    /// Remove all targets
    void clear();

    virtual void init() override;

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
    Data<float> d_showObjectScale;
    Data<defaulttype::RGBAColor> d_color;

protected:
    DistanceFromTargetMapping();
    virtual ~DistanceFromTargetMapping();

    SparseMatrixEigen jacobian;                      ///< Jacobian of the mapping
    helper::vector<defaulttype::BaseMatrix*> baseMatrices;   ///< Jacobian of the mapping, in a vector
    SparseKMatrixEigen K;  ///< Assembled geometric stiffness matrix
    helper::vector<Direction> directions;                         ///< Unit vectors in the directions of the lines
    helper::vector< Real > invlengths;                          ///< inverse of current distances. Null represents the infinity (null distance)

    /// r=b-a only for position (eventual rotation, affine transform... remains null)
    void computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b );

    virtual void updateForceMask() override;

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_DistanceFromTargetMapping_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< defaulttype::Vec1dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< defaulttype::Rigid3dTypes, defaulttype::Vec1dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< defaulttype::Vec3fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< defaulttype::Vec1fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< defaulttype::Rigid3fTypes, defaulttype::Vec1fTypes >;
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
