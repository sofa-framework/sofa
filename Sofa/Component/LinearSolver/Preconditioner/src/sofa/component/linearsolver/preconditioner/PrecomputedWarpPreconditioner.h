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
#include <sofa/component/linearsolver/preconditioner/config.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/helper/map.h>
#include <cmath>
#include <fstream>

namespace sofa::component::linearsolver::preconditioner
{

template<class TDataTypes>
class PrecomputedWarpPreconditionerInternalData
{
public :
    typedef typename TDataTypes::Coord Coord;
    typedef typename Coord::value_type Real;
    typedef linearalgebra::FullMatrix<Real> TBaseMatrix;
    typedef linearalgebra::FullVector<Real> TBaseVector;

    linearalgebra::SparseMatrix<Real> JR;
    linearalgebra::FullMatrix<Real> JRMinv;
    linearalgebra::FullMatrix<Real>* MinvPtr;
    std::vector<int> idActiveDofs;
    std::vector<int> invActiveDofs;
    bool shared;
    PrecomputedWarpPreconditionerInternalData()
        : MinvPtr(new linearalgebra::FullMatrix<Real>), shared(false)
    {
    }

    ~PrecomputedWarpPreconditionerInternalData()
    {
        if (!shared && MinvPtr!=nullptr) delete MinvPtr;
    }

    void setMinv(linearalgebra::FullMatrix<Real>* m, bool shared = true)
    {
        if (!this->shared && MinvPtr!=nullptr) delete this->MinvPtr;
        this->MinvPtr = m;
        this->shared = shared;
    }

    static linearalgebra::FullMatrix<Real>* getSharedMatrix(const std::string& name)
    {
        static std::map< std::string, linearalgebra::FullMatrix<Real> > matrices;
        return &(matrices[name]);
    }

    void readMinvFomFile(std::ifstream & compFileIn)
    {
        compFileIn.read((char*) (*MinvPtr)[0], MinvPtr->colSize() * MinvPtr->rowSize() * sizeof(Real));
    }

    void writeMinvFomFile(std::ofstream & compFileOut)
    {
        compFileOut.write((char*) (*MinvPtr)[0], MinvPtr->colSize() * MinvPtr->rowSize() * sizeof(Real));
    }
};

/// Linear system solver based on a precomputed inverse matrix, wrapped by a per-node rotation matrix
template<class TDataTypes>
class PrecomputedWarpPreconditioner : public sofa::component::linearsolver::MatrixLinearSolver<linearalgebra::CompressedRowSparseMatrix<typename TDataTypes::Real>,typename PrecomputedWarpPreconditionerInternalData<TDataTypes>::TBaseVector>
{
public:
    typedef typename TDataTypes::Real Real;
    typedef linearalgebra::CompressedRowSparseMatrix<Real> TMatrix;
    typedef typename PrecomputedWarpPreconditionerInternalData<TDataTypes>::TBaseVector TVector;
    typedef typename PrecomputedWarpPreconditionerInternalData<TDataTypes>::TBaseMatrix TBaseMatrix;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(PrecomputedWarpPreconditioner,TDataTypes),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));
    typedef TDataTypes DataTypes;
    typedef typename TDataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename TDataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename sofa::core::behavior::MechanicalState<DataTypes> MState;

    typedef sofa::type::MatNoInit<3, 3, Real> Transformation;

    Data<bool> d_jmjt_twostep; ///< Use two step algorithm to compute JMinvJt
    Data<bool> d_use_file; ///< Dump system matrix in a file
    Data<bool> d_share_matrix; ///< Share the compliance matrix in memory if they are related to the same file (WARNING: might require to reload Sofa when opening a new scene...)
    SingleLink<PrecomputedWarpPreconditioner, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_linearSolver; ///< Link towards the linear solver used to precompute the first matrix
    Data<bool> d_use_rotations; ///< Use Rotations around the preconditioner
    Data<double> d_draw_rotations_scale; ///< Scale rotations in draw function

    MState * mstate;

protected:
    PrecomputedWarpPreconditioner();

    void checkLinearSystem() override;

public:
    void solve (TMatrix& M, TVector& x, TVector& b) override;
    void invert(TMatrix& M) override;
    bool addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact) override;
    void draw(const core::visual::VisualParams* vparams) override;
    void init() override;
    void loadMatrix(TMatrix& M);

    TBaseMatrix * getSystemMatrixInv()
    {
        return internalData.MinvPtr;
    }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<sofa::core::behavior::MechanicalState<TDataTypes> *>(context->getMechanicalState()) == nullptr) {
            arg->logError(std::string("No mechanical state with the datatype '") + TDataTypes::Name() + "' found in the context node.");
            return false;
        }
        return sofa::core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

protected :
    TVector R;
    TVector T;

    std::vector<bool> isActiveDofs;
    PrecomputedWarpPreconditionerInternalData<TDataTypes> internalData;

    void rotateConstraints();
    void loadMatrixWithCholeskyDecomposition(TMatrix& M);
    void loadMatrixWithSolver();

    template<class JMatrix>
    void ComputeResult(linearalgebra::BaseMatrix * result,JMatrix& J, float fact);


    template<class JMatrix>
    void computeActiveDofs(JMatrix& J);

    bool first;
    bool _rotate;
    bool usePrecond;
    double init_mFact;
    double init_bFact;
    double init_kFact;
    double dt;
    double factInt;
    unsigned systemSize;
    unsigned dof_on_node;
    unsigned nb_dofs;
    unsigned matrixSize;

};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_PRECOMPUTEDWARPPRECONDITIONER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API PrecomputedWarpPreconditioner< defaulttype::Vec3Types >;
#endif // !defined(SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_PRECOMPUTEDWARPPRECONDITIONER_CPP)

} // namespace sofa::component::linearsolver::preconditioner
