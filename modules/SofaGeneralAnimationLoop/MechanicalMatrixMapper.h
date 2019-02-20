/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2018 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ANIMATIONLOOP_MECHANICALMATRIXMAPPER_H
#define SOFA_COMPONENT_ANIMATIONLOOP_MECHANICALMATRIXMAPPER_H

#include <sofa/core/behavior/MixedInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/core/MechanicalParams.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <SofaBaseLinearSolver/DefaultMultiMatrixAccessor.h>
#include <SofaMiscMapping/config.h>

#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/defaulttype/BaseMatrix.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{


class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalAccumulateJacobian : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalAccumulateJacobian(const core::ConstraintParams* _cparams, core::MultiMatrixDerivId _res)
        : simulation::BaseMechanicalVisitor(_cparams)
        , res(_res)
        , cparams(_cparams)
    {

    }

    virtual void bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map)
    {
        ctime_t t0 = begin(node, map);
        map->applyJT(cparams, res, res);
        end(node, map, t0);
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAccumulateJacobian"; }

    virtual bool isThreadSafe() const
    {
        return false;
    }
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

protected:
    core::MultiMatrixDerivId res;
    const sofa::core::ConstraintParams *cparams;
};

using sofa::core::objectmodel::BaseObject ;
using sofa::component::linearsolver::CompressedRowSparseMatrix ;
using sofa::core::behavior::MixedInteractionForceField ;
using sofa::core::behavior::BaseForceField ;
using sofa::core::behavior::BaseMass ;
using sofa::core::behavior::BaseMechanicalState ;
using sofa::core::behavior::MultiMatrixAccessor ;
using sofa::component::linearsolver::DefaultMultiMatrixAccessor ;
using sofa::defaulttype::BaseMatrix ;
using sofa::core::MechanicalParams ;
using sofa::core::objectmodel::ComponentState ;


/**
 * \brief This component allows to map mechanical matrices (Stiffness, Mass) through a mapping.
 *
 * This is needed in SOFA scenes having these two following particularities:
 *  - There are using a direct solver (e.g. SparseLDLSolver) that, unlike
 *    iterative solvers, need to build the mechanical matrices.
 *  - They involves ForceFields that implement addKToMatrix (i.e. that compute internal forces such as e.g. TetrahedronFEMForceField,
 *    TetrahedronHyperElasticityFEMForceField, but not ConstantForceField which only contributes to the right-hand side) and that
 *    ARE USED UNDER mappings.
 * Without this component, such a scene either crashes or gives unlogical behaviour.
 *
 * The component supports the case of subsetMultiMappings that map from one to two mechanical objects.
 * An example using this component can be found in examples/Components/animationLoop/MechanicalMatrixMapper.pyscn
*/
template<typename TDataTypes1, typename TDataTypes2>
class MechanicalMatrixMapper : public MixedInteractionForceField<TDataTypes1, TDataTypes2>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(MechanicalMatrixMapper, TDataTypes1, TDataTypes2), SOFA_TEMPLATE2(MixedInteractionForceField, TDataTypes1, TDataTypes2));

    typedef MixedInteractionForceField<TDataTypes1, TDataTypes2> Inherit;

    // Vec3
    typedef TDataTypes1 DataTypes1;
    typedef typename DataTypes1::VecCoord VecCoord1;
    typedef typename DataTypes1::VecDeriv VecDeriv1;
    typedef typename DataTypes1::Coord    Coord1;
    typedef typename DataTypes1::Deriv    Deriv1;
    typedef typename DataTypes1::Real     Real1;
    typedef typename DataTypes1::MatrixDeriv MatrixDeriv1;
    typedef Data<MatrixDeriv1>  DataMatrixDeriv1;
    typedef typename DataTypes1::MatrixDeriv::RowConstIterator MatrixDeriv1RowConstIterator;
    typedef typename DataTypes1::MatrixDeriv::ColConstIterator MatrixDeriv1ColConstIterator;
    static const unsigned int DerivSize1 = Deriv1::total_size;
    typedef Data<VecCoord1>    DataVecCoord1;
    typedef Data<VecDeriv1>    DataVecDeriv1;

    // Rigid
    typedef TDataTypes2 DataTypes2;
    typedef typename DataTypes2::VecCoord VecCoord2;
    typedef typename DataTypes2::VecDeriv VecDeriv2;
    typedef typename DataTypes2::Coord    Coord2;
    typedef typename DataTypes2::Deriv    Deriv2;
    typedef typename DataTypes2::Real     Real2;
    typedef typename DataTypes2::MatrixDeriv MatrixDeriv2;
    typedef Data<MatrixDeriv2>  DataMatrixDeriv2;
    typedef typename DataTypes2::MatrixDeriv::RowConstIterator MatrixDeriv2RowConstIterator;
    typedef typename DataTypes2::MatrixDeriv::ColConstIterator MatrixDeriv2ColConstIterator;
    static const unsigned int DerivSize2 = Deriv2::total_size;
    typedef Data<VecCoord2>    DataVecCoord2;
    typedef Data<VecDeriv2>    DataVecDeriv2;

protected:

    Data<helper::vector<std::string>> d_forceFieldList;
    SingleLink < MechanicalMatrixMapper<DataTypes1, DataTypes2>, sofa::simulation::Node , BaseLink::FLAG_STOREPATH > l_nodeToParse;
    Data <bool> d_stopAtNodeToParse;
    Data <bool> d_skipJ1tKJ1;
    Data <bool> d_skipJ2tKJ2;
    SingleLink < MechanicalMatrixMapper<DataTypes1, DataTypes2>, sofa::core::behavior::BaseMechanicalState , BaseLink::FLAG_NONE > l_mechanicalState;
    SingleLink < MechanicalMatrixMapper<DataTypes1, DataTypes2>, sofa::core::behavior::BaseMass , BaseLink::FLAG_NONE > l_mappedMass;
    MultiLink  < MechanicalMatrixMapper<DataTypes1, DataTypes2>, sofa::core::behavior::BaseForceField, BaseLink::FLAG_NONE > l_forceField;


    size_t m_nbInteractionForceFields;

    MechanicalMatrixMapper() ;

public:

    ////////////////////////// Inherited from BaseObject //////////////////////
    virtual void init() override;
    ///////////////////////////////////////////////////////////////////////////

    ////////////////////////// Inherited from ForceField //////////////////////
    virtual void addForce(const MechanicalParams* mparams,
                          DataVecDeriv1& f1,
                          DataVecDeriv2& f2,
                          const DataVecCoord1& x1,
                          const DataVecCoord2& x2,
                          const DataVecDeriv1& v1,
                          const DataVecDeriv2& v2) override;

    virtual void addDForce(const MechanicalParams* mparams,
                           DataVecDeriv1& df1,
                           DataVecDeriv2& df2,
                           const DataVecDeriv1& dx1,
                           const DataVecDeriv2& dx2) override;

    virtual void addKToMatrix(const MechanicalParams* mparams,
                              const MultiMatrixAccessor* matrix ) override;

    virtual double getPotentialEnergy(const MechanicalParams* mparams,
                                      const DataVecCoord1& x1, const DataVecCoord2& x2) const override;
    ///////////////////////////////////////////////////////////////////////////


    /**
     * \brief Walk recursively through a node and its children linking with their forcefield
     * \param node : from which node it start
     * \param massName : won't link with given mass component
     */
    void parseNode(sofa::simulation::Node *node ,std::string massName);


protected:
    /**
     * \brief This method computes the mapping jacobian.
     *
    */
    void accumulateJacobians(const MechanicalParams* mparams);

    /**
     * \brief This method fills the jacobian matrix (of the mapping) with identity blocks on the provided list of nodes(dofs)
     *
     *
    */
    virtual void buildIdentityBlocksInJacobian(core::behavior::BaseMechanicalState* mstate, sofa::core::MatrixDerivId Id);

    /**
     * \brief This method encapsulates the jacobian accumulation to allow for specific optimisations in child classes
     *
     *
    */
    virtual void accumulateJacobiansOptimized(const MechanicalParams* mparams);

    /**
     * \brief This method adds the mass matrix to the system
     *
     *
    */
    virtual void addMassToSystem(const MechanicalParams* mparams, const DefaultMultiMatrixAccessor* KAccessor);

    /**
     * \brief This method does not do anything in this class but is used in child classes.
     *
     *
    */
    virtual void addPrecomputedMassToSystem(const MechanicalParams* mparams,const unsigned int mstateSize,const Eigen::SparseMatrix<double> &Jeig, Eigen::SparseMatrix<double>& JtKJeig);

    /**
     * \brief This method performs the copy of the jacobian matrix of the first  mstate to the eigen sparse format
     *
     *
    */
    virtual void optimizeAndCopyMappingJacobianToEigenFormat1(const typename DataTypes1::MatrixDeriv& J, Eigen::SparseMatrix<double>& Jeig);

    /**
     * \brief This method performs the copy of the jacobian matrix of the second mstate (not mandatory) to the eigen sparse format
     *
     *
    */
    virtual void optimizeAndCopyMappingJacobianToEigenFormat2(const typename DataTypes2::MatrixDeriv& J, Eigen::SparseMatrix<double>& Jeig);

    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using MixedInteractionForceField<TDataTypes1, TDataTypes2>::f_printLog ;
    using MixedInteractionForceField<TDataTypes1, TDataTypes2>::mstate1 ;
    using MixedInteractionForceField<TDataTypes1, TDataTypes2>::mstate2 ;
    using MixedInteractionForceField<TDataTypes1, TDataTypes2>::getContext ;
    using BaseObject::m_componentstate ;
    ////////////////////////////////////////////////////////////////////////////

};

#if !defined(SOFA_COMPONENT_ANIMATIONLOOP_MECHANICALMATRIXMAPPER_CPP)
extern template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<defaulttype::Vec3Types, defaulttype::Rigid3Types>;
extern template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<defaulttype::Vec3Types, defaulttype::Vec3Types>;
extern template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<defaulttype::Vec1Types, defaulttype::Rigid3Types>;
extern template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<defaulttype::Vec1Types, defaulttype::Vec1Types>;
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ANIMATIONLOOP_MECHANICALMATRIXMAPPER_H
