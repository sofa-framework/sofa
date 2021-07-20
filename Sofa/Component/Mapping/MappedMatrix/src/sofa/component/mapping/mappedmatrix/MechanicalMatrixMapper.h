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

#include <sofa/component/mapping/mappedmatrix/config.h>

#ifndef SOFA_BUILD_SOFA_COMPONENT_MAPPING_MAPPEDMATRIX
SOFA_DEPRECATED_HEADER_NOT_REPLACED("v23.06", "v23.12")
#endif

#include <sofa/core/behavior/MixedInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/core/behavior/DefaultMultiMatrixAccessor.h>

#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/linearalgebra/SparseMatrixProduct[EigenSparseMatrix].h>

#include <sofa/simulation/fwd.h>

namespace sofa::component::mapping::mappedmatrix
{

using sofa::core::objectmodel::BaseObject ;
using sofa::linearalgebra::CompressedRowSparseMatrix ;
using sofa::core::behavior::MixedInteractionForceField ;
using sofa::core::behavior::BaseForceField ;
using sofa::core::behavior::BaseMass ;
using sofa::core::behavior::BaseMechanicalState ;
using sofa::core::behavior::MultiMatrixAccessor ;
using sofa::core::behavior::DefaultMultiMatrixAccessor;
using sofa::linearalgebra::BaseMatrix ;
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
 * An example using this component can be found in examples/Component/Mapping/MappedMatrix/MechanicalMatrixMapper.pyscn
*/
template<typename TDataTypes1, typename TDataTypes2>
// SOFA_ATTRIBUTE_DEPRECATED("v23.06", "v23.12", "Matrix mapping is now supported automatically. Therefore, MechanicalMatrixMapper is no longer necessary.")
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

    Data<bool> d_yesIKnowMatrixMappingIsSupportedAutomatically;

    Data<type::vector<std::string>> d_forceFieldList; ///< List of ForceField Names to work on (by default will take all)
    SingleLink < MechanicalMatrixMapper<DataTypes1, DataTypes2>, sofa::simulation::Node , BaseLink::FLAG_STOREPATH > l_nodeToParse;
    Data <bool> d_stopAtNodeToParse; ///< Boolean to choose whether forceFields in children Nodes of NodeToParse should be considered.
    Data <bool> d_skipJ1tKJ1; ///< Boolean to choose whether to skip J1tKJ1 to avoid 2 contributions, in case 2 MechanicalMatrixMapper are used
    Data <bool> d_skipJ2tKJ2; ///< Boolean to choose whether to skip J2tKJ2 to avoid 2 contributions, in case 2 MechanicalMatrixMapper are used
    Data <bool> d_fastMatrixProduct; ///< If true, an accelerated method to compute matrix products based on the pre-computation of the matrices intersection is used. Regular matrix product otherwise.
    Data <bool> d_parallelTasks; ///< Execute some tasks in parallel for better performances
    Data <bool> d_forceFieldAndMass; ///< If true, allows forceField and mass to be in the same component.
    SingleLink < MechanicalMatrixMapper<DataTypes1, DataTypes2>, sofa::core::behavior::BaseMechanicalState , BaseLink::FLAG_NONE > l_mechanicalState;
    SingleLink < MechanicalMatrixMapper<DataTypes1, DataTypes2>, sofa::core::behavior::BaseMass , BaseLink::FLAG_NONE > l_mappedMass;
    MultiLink  < MechanicalMatrixMapper<DataTypes1, DataTypes2>, sofa::core::behavior::BaseForceField, BaseLink::FLAG_NONE > l_forceField;


    unsigned int m_nbColsJ1, m_nbColsJ2;
    Eigen::SparseMatrix<double> m_J1eig;
    Eigen::SparseMatrix<double> m_J2eig;
    linearalgebra::SparseMatrixProduct<Eigen::SparseMatrix<double> > m_product_J1tKJ1;
    linearalgebra::SparseMatrixProduct<Eigen::SparseMatrix<double> > m_product_J2tKJ2;
    linearalgebra::SparseMatrixProduct<Eigen::SparseMatrix<double> > m_product_J1tKJ2;
    linearalgebra::SparseMatrixProduct<Eigen::SparseMatrix<double> > m_product_J2tKJ1;

    /// The matrix product J^T * K is required in more than one computation. This structure stores the result
    /// and says if the product has already been computed or not.
    struct JtKMatrixProduct
    {
        /// If the fast sparse matrix product is used
        linearalgebra::SparseMatrixProduct<Eigen::SparseMatrix<double> > product;
        /// If the regular sparse matrix product is used
        Eigen::SparseMatrix<double> matrix;

        bool isComputed { false };
    };

    JtKMatrixProduct m_product_J1tK; /// J1^T * K
    JtKMatrixProduct m_product_J2tK; /// J2^T * K

    Eigen::SparseMatrix<double> m_J1tKJ1eigen;
    Eigen::SparseMatrix<double> m_J2tKJ2eigen;
    Eigen::SparseMatrix<double> m_J1tKJ2eigen;
    Eigen::SparseMatrix<double> m_J2tKJ1eigen;

    /// Compute J1^T * K * J2 using the accelerated sparse matrix products
    static void computeMatrixProduct(
        bool fastProduct,
        JtKMatrixProduct& product_1,
        linearalgebra::SparseMatrixProduct<Eigen::SparseMatrix<double> >& product_2,
        const Eigen::SparseMatrix<double>* J1, const Eigen::SparseMatrix<double>* J2,
        const Eigen::SparseMatrix<double>* K,
        Eigen::SparseMatrix<double>*& output);

    unsigned int m_fullMatrixSize;
    size_t m_nbInteractionForceFields;

    int m_topologyRevision { -1 };

    MechanicalMatrixMapper() ;

public:

    ////////////////////////// Inherited from BaseObject //////////////////////
    void init() override;
    void bwdInit() override;
    ///////////////////////////////////////////////////////////////////////////

    ////////////////////////// Inherited from ForceField //////////////////////
    void addForce(const MechanicalParams* mparams,
                          DataVecDeriv1& f1,
                          DataVecDeriv2& f2,
                          const DataVecCoord1& x1,
                          const DataVecCoord2& x2,
                          const DataVecDeriv1& v1,
                          const DataVecDeriv2& v2) override;

    void addDForce(const MechanicalParams* mparams,
                           DataVecDeriv1& df1,
                           DataVecDeriv2& df2,
                           const DataVecDeriv1& dx1,
                           const DataVecDeriv2& dx2) override;

    void addKToMatrix(const MechanicalParams* mparams,
                              const MultiMatrixAccessor* matrix ) override;

    SReal getPotentialEnergy(const MechanicalParams* mparams,
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

public:
    void parse(core::objectmodel::BaseObjectDescription* arg) override;

protected:
    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using MixedInteractionForceField<TDataTypes1, TDataTypes2>::f_printLog ;
    using MixedInteractionForceField<TDataTypes1, TDataTypes2>::mstate1 ;
    using MixedInteractionForceField<TDataTypes1, TDataTypes2>::mstate2 ;
    using MixedInteractionForceField<TDataTypes1, TDataTypes2>::getContext ;
    ////////////////////////////////////////////////////////////////////////////

    class JacobianTask;
};

#if !defined(SOFA_COMPONENT_ANIMATIONLOOP_MECHANICALMATRIXMAPPER_CPP)
extern template class SOFA_COMPONENT_MAPPING_MAPPEDMATRIX_API MechanicalMatrixMapper<defaulttype::Rigid3Types, defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_MAPPING_MAPPEDMATRIX_API MechanicalMatrixMapper<defaulttype::Vec3Types, defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_MAPPING_MAPPEDMATRIX_API MechanicalMatrixMapper<defaulttype::Vec3Types, defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MAPPING_MAPPEDMATRIX_API MechanicalMatrixMapper<defaulttype::Vec1Types, defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_MAPPING_MAPPEDMATRIX_API MechanicalMatrixMapper<defaulttype::Vec1Types, defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MAPPING_MAPPEDMATRIX_API MechanicalMatrixMapper<defaulttype::Vec1Types, defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_MAPPING_MAPPEDMATRIX_API MechanicalMatrixMapper<defaulttype::Rigid3Types, defaulttype::Vec1Types>;
#endif

} // namespace sofa::component::mapping::mappedmatrix
