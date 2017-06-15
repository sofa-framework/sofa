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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectDirectionConstraint_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectDirectionConstraint_H
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
#include <sofa/defaulttype/Mat.h>
#include <SofaBaseTopology/TopologySubsetData.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <set>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class ProjectDirectionConstraintInternalData
{

};

/** Project particles to an affine straight line going through the particle original position.
*/
template <class DataTypes>
class ProjectDirectionConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ProjectDirectionConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

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
    typedef helper::vector<unsigned int> Indices;
    typedef sofa::defaulttype::Vector3 Vector3;
    typedef sofa::component::topology::PointSubsetData< Indices > IndexSubsetData;
    typedef linearsolver::EigenBaseSparseMatrix<SReal> BaseSparseMatrix;
    typedef linearsolver::EigenSparseMatrix<DataTypes,DataTypes> SparseMatrix;
    typedef typename SparseMatrix::Block Block;                                       ///< projection matrix of a particle displacement to the plane
    enum {bsize=SparseMatrix::Nin};                                                   ///< size of a block


protected:
    ProjectDirectionConstraint();

    virtual ~ProjectDirectionConstraint();

public:
    IndexSubsetData f_indices;  ///< the particles to project
    Data<SReal> f_drawSize;    ///< The size of the square used to display the constrained particles
    Data<CPos> f_direction;    ///< The direction of the line. Will be normalized by init()


protected:
    ProjectDirectionConstraintInternalData<DataTypes>* data;
    friend class ProjectDirectionConstraintInternalData<DataTypes>;

    helper::vector<CPos> m_origin;


public:
    void clearConstraints();
    void addConstraint(unsigned int index);
    void removeConstraint(unsigned int index);

    // -- Constraint interface
    virtual void init();
    virtual void reinit();

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData);
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData);
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData);
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData);

    using core::behavior::ProjectiveConstraintSet<DataTypes>::applyConstraint;
    void applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset);
    void applyConstraint(defaulttype::BaseVector *vect, unsigned int offset);

    /// Project the the given matrix (Experimental API, see the spec in sofa::core::behavior::BaseProjectiveConstraintSet).
    virtual void projectMatrix( sofa::defaulttype::BaseMatrix* /*M*/, unsigned /*offset*/ );


    virtual void draw(const core::visual::VisualParams* vparams);


    class FCPointHandler : public component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, Indices >
    {
    public:
        typedef typename ProjectDirectionConstraint<DataTypes>::Indices Indices;
        typedef sofa::core::topology::Point Point;
        FCPointHandler(ProjectDirectionConstraint<DataTypes>* _fc, component::topology::PointSubsetData<Indices>* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, Indices >(_data), fc(_fc) {}


        using component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, Indices >::applyDestroyFunction;
        void applyDestroyFunction(unsigned int /*index*/, core::objectmodel::Data<value_type>& /*T*/);


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        ProjectDirectionConstraint<DataTypes> *fc;
    };

protected :
    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;

    /// Handler for subset Data
    FCPointHandler* pointHandler;

    SparseMatrix jacobian; ///< projection matrix in local state
    SparseMatrix J;        ///< auxiliary variable
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectDirectionConstraint_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Vec2dTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Vec1dTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Vec6dTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Rigid3dTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Vec2fTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Vec1fTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Vec6fTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Rigid3fTypes>;
//extern template class SOFA_BOUNDARY_CONDITION_API ProjectDirectionConstraint<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectDirectionConstraint_H
