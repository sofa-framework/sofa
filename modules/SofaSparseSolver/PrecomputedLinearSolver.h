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
#ifndef SOFA_COMPONENT_LINEARSOLVER_PrecomputedLinearSolver_H
#define SOFA_COMPONENT_LINEARSOLVER_PrecomputedLinearSolver_H
#include "config.h"

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <sofa/helper/map.h>
#include <math.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class TMatrix, class TVector>
class PrecomputedLinearSolverInternalData
{
public :
    typedef typename TMatrix::Real Real;
    typedef FullMatrix<Real> TBaseMatrix ;

    FullMatrix<Real> JMinv;
    FullMatrix<Real> Minv;
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

    Data<bool> jmjt_twostep;
    Data<bool> f_verbose;
    Data<bool> use_file;
    Data<int> init_MaxIter;
    Data<double> init_Tolerance;
    Data<double> init_Threshold;

    PrecomputedLinearSolver();
    void solve (TMatrix& M, TVector& x, TVector& b);
    void invert(TMatrix& M);
    void setSystemMBKMatrix(const core::MechanicalParams* mparams);
    void loadMatrix(TMatrix& M);
#ifdef SOFA_HAVE_CSPARSE
    void loadMatrixWithCSparse(TMatrix& M);
#endif
    bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact);


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const PrecomputedLinearSolver<TMatrix,TVector>* = NULL)
    {
        return TVector::Name();
    }

    TBaseMatrix * getSystemMatrixInv()
    {
        return &internalData.Minv;
    }

protected :
    template<class JMatrix>
    void ComputeResult(defaulttype::BaseMatrix * result,JMatrix& J, float fact);

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

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
