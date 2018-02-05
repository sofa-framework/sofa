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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPointConstraint_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPointConstraint_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
//#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/TopologySubsetData.h>
#include <SofaEigen2Solver/EigenBaseSparseMatrix.h>
#include <set>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class ProjectToPointConstraintInternalData
{

};

/** Attach given particles to their initial positions.
 * Contrary to FixedConstraint, this one stops the particles even if they have a non-null initial velocity.
 * @sa FixedConstraint
*/
template <class DataTypes>
class ProjectToPointConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ProjectToPointConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;
    typedef sofa::defaulttype::Vector3 Vector3;

protected:
    ProjectToPointConstraint();

    virtual ~ProjectToPointConstraint();

public:
    SetIndex f_indices;    ///< the indices of the points to project to the target
    Data<Coord> f_point;    ///< the target of the projection
    Data<bool> f_fixAll;    ///< to project all the points, rather than those listed in f_indices
    Data<SReal> f_drawSize;


protected:
    ProjectToPointConstraintInternalData<DataTypes>* data;
    friend class ProjectToPointConstraintInternalData<DataTypes>;


public:
    void clearConstraints();
    void addConstraint(unsigned int index);
    void removeConstraint(unsigned int index);

    // -- Constraint interface
    virtual void init() override;
    virtual void reinit() override;

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    using core::behavior::ProjectiveConstraintSet<DataTypes>::applyConstraint;
    void applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset);
    void applyConstraint(defaulttype::BaseVector *vect, unsigned int offset);

    /** Project the the given matrix (Experimental API).
      Replace M with PMP, where P is the projection matrix corresponding to the projectResponse method, shifted by the given offset, i.e. P is the identity matrix with a block on the diagonal replaced by the projection matrix.
      */
    virtual void projectMatrix( sofa::defaulttype::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;


    virtual void draw(const core::visual::VisualParams* vparams) override;

    bool fixAllDOFs() const { return f_fixAll.getValue(); }

    class FCPointHandler : public component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >
    {
    public:
        typedef typename ProjectToPointConstraint<DataTypes>::SetIndexArray SetIndexArray;
        typedef sofa::core::topology::Point Point;
        FCPointHandler(ProjectToPointConstraint<DataTypes>* _fc, component::topology::PointSubsetData<SetIndexArray>* _data)
            : component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >(_data), fc(_fc) {}


        using component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >::applyDestroyFunction;
        void applyDestroyFunction(unsigned int /*index*/, core::objectmodel::Data<value_type>& /*T*/);


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        ProjectToPointConstraint<DataTypes> *fc;
    };

protected :
    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;

    /// Handler for subset Data
    FCPointHandler* pointHandler;

    /// Matrix used in getJ
//    linearsolver::EigenBaseSparseMatrix<SReal> jacobian;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPointConstraint_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Vec1dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Vec6dTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Rigid3dTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Vec1fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Vec6fTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Rigid3fTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif
