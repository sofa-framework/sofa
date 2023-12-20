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
#include <sofa/component/constraint/projective/config.h>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/vector.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <set>

namespace sofa::component::constraint::projective
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class PlaneProjectiveConstraintInternalData
{

};

/** Project particles to an affine plane.
  @author Francois Faure, 2012
  @todo Optimized versions for planes parallel to the main directions
*/
template <class DataTypes>
class PlaneProjectiveConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PlaneProjectiveConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    using Index = sofa::Index;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos CPos;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef type::vector<Index> Indices;
    typedef sofa::core::topology::TopologySubsetIndices IndexSubsetData;
    typedef linearalgebra::EigenBaseSparseMatrix<SReal> BaseSparseMatrix;
    typedef linearalgebra::EigenSparseMatrix<DataTypes,DataTypes> SparseMatrix;
    typedef typename SparseMatrix::Block Block;                                       ///< projection matrix of a particle displacement to the plane
    enum {bsize=SparseMatrix::Nin};                                                   ///< size of a block

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);

protected:
    PlaneProjectiveConstraint();

    virtual ~PlaneProjectiveConstraint();

public:
    IndexSubsetData f_indices;  ///< the particles to project
    Data<CPos> f_origin;       ///< A point in the plane
    Data<CPos> f_normal;       ///< The normal to the plane. Will be normalized by init().
    Data<SReal> f_drawSize;    ///< The size of the display of the constrained particles

    /// Link to be set to the topology container in the component graph.
    SingleLink<PlaneProjectiveConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    PlaneProjectiveConstraintInternalData<DataTypes>* data;
    friend class PlaneProjectiveConstraintInternalData<DataTypes>;


public:
    void clearConstraints();
    void addConstraint(Index index);
    void removeConstraint(Index index);

    // -- Constraint interface
    void init() override;
    void reinit() override;

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    void applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    /** Project the given matrix (Experimental API).
      Replace M with PMP, where P is the projection matrix corresponding to the projectResponse method, shifted by the given offset, i.e. P is the identity matrix with a block on the diagonal replaced by the projection matrix.
      */
    void projectMatrix( sofa::linearalgebra::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;


    void draw(const core::visual::VisualParams* vparams) override;

protected :

    SparseMatrix jacobian; ///< projection matrix in local state
    SparseMatrix J;        ///< auxiliary variable
};


#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PlaneProjectiveConstraint_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PlaneProjectiveConstraint<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PlaneProjectiveConstraint<defaulttype::Vec2Types>;

#endif

} // namespace sofa::component::constraint::projective

