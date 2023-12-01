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
#include <sofa/component/linearsolver/direct/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/helper/map.h>
#include <cmath>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <fstream>

namespace sofa::component::linearsolver::direct
{

template<class TMatrix, class TVector>
class PrecomputedLinearSolverInternalData
{
public :
    typedef typename TMatrix::Real Real;
    typedef sofa::linearalgebra::FullMatrix<Real> TBaseMatrix ;

    sofa::linearalgebra::FullMatrix<Real> JMinv;
    sofa::linearalgebra::FullMatrix<Real> Minv;
    std::vector<int> idActiveDofs;
    std::vector<int> invActiveDofs;

    bool readFile(const char * filename,unsigned systemSize)
    {
        std::ifstream compFileIn(filename, std::ifstream::binary);

        if(compFileIn.good())
        {
            msg_info("PrecomputedLInearSolverInternalData") << "file '" << filename << "' with compliance being loaded." ;
            compFileIn.read((char*) Minv[0], systemSize * systemSize * sizeof(Real));
            compFileIn.close();
            return true;
        }
        return false;
    }

    void writeFile(const char * filename,unsigned systemSize)
    {
        std::ofstream compFileOut(filename, std::fstream::out | std::fstream::binary);
        compFileOut.write((char*) Minv[0], systemSize * systemSize*sizeof(Real));
        compFileOut.close();
    }
};

/// Linear system solver based on a precomputed inverse matrix
template<class TMatrix, class TVector>
class PrecomputedLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(PrecomputedLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;
    typedef typename TMatrix::Real Real;
    typedef typename PrecomputedLinearSolverInternalData<TMatrix,TVector>::TBaseMatrix TBaseMatrix;

    Data<bool> jmjt_twostep; ///< Use two step algorithm to compute JMinvJt
    Data<bool> use_file; ///< Dump system matrix in a file
    Data<double> init_Tolerance;

    SOFA_ATTRIBUTE_DEPRECATED__SOLVER_DIRECT_VERBOSEDATA()
    Data<bool> f_verbose; ///< Dump system state at each iteration

    PrecomputedLinearSolver();
    void solve (TMatrix& M, TVector& x, TVector& b) override;
    void invert(TMatrix& M) override;
    void setSystemMBKMatrix(const core::MechanicalParams* mparams) override;
    void loadMatrix(TMatrix& M);
    void loadMatrixWithCholeskyDecomposition(TMatrix& M);
    bool addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact) override;

    /// Returns the sofa template name. By default the name of the c++ class signature is exposed...
    /// so we need to override that by implementing GetCustomTemplateName() function
    /// More details on the name customization infrastructure is in NameDecoder.h
    static const std::string GetCustomTemplateName()
    {
        return TVector::Name();
    }

    TBaseMatrix * getSystemMatrixInv()
    {
        return &internalData.Minv;
    }

    void parse(core::objectmodel::BaseObjectDescription *arg) override;

protected :
    template<class JMatrix>
    void ComputeResult(linearalgebra::BaseMatrix * result,JMatrix& J, SReal fact);

    PrecomputedLinearSolverInternalData<TMatrix,TVector> internalData;

    template<class JMatrix>
    void computeActiveDofs(JMatrix& J);


private :
    bool first;
    unsigned systemSize;
    double dt;
    double factInt;
    std::vector<bool> isActiveDofs;
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_PRECOMPUTEDLINEARSOLVER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API PrecomputedLinearSolver< linearalgebra::CompressedRowSparseMatrix<SReal> , linearalgebra::FullVector<SReal> >;
#endif

} // namespace sofa::component::linearsolver::direct
