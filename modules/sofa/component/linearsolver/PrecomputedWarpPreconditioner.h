/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_PrecomputedWarpPreconditioner_H
#define SOFA_COMPONENT_LINEARSOLVER_PrecomputedWarpPreconditioner_H

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/helper/map.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::core;
using namespace sofa::defaulttype;

using namespace sofa::core::behavior;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;

using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

template<class TDataTypes>
class PrecomputedWarpPreconditionerInternalData
{
public :
    typedef typename TDataTypes::Coord Coord;
    typedef typename Coord::value_type Real;
    typedef FullMatrix<Real> TBaseMatrix;
    typedef FullVector<Real> TBaseVector;

    SparseMatrix<Real> JR;
    FullMatrix<Real> JRMinv;
    FullMatrix<Real>* MinvPtr;
    bool shared;
    PrecomputedWarpPreconditionerInternalData()
        : MinvPtr(new FullMatrix<Real>), shared(false)
    {
    }

    ~PrecomputedWarpPreconditionerInternalData()
    {
        if (!shared && MinvPtr!=NULL) delete MinvPtr;
    }

    void setMinv(FullMatrix<Real>* m, bool shared = true)
    {
        if (!this->shared && MinvPtr!=NULL) delete this->MinvPtr;
        this->MinvPtr = m;
        this->shared = shared;
    }

    static FullMatrix<Real>* getSharedMatrix(const std::string& name)
    {
        static std::map< std::string,FullMatrix<Real> > matrices;
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
class PrecomputedWarpPreconditioner : public sofa::component::linearsolver::MatrixLinearSolver<CompressedRowSparseMatrix<typename TDataTypes::Real>,typename PrecomputedWarpPreconditionerInternalData<TDataTypes>::TBaseVector>
{
public:
    typedef typename TDataTypes::Real Real;
    typedef CompressedRowSparseMatrix<Real> TMatrix;
    typedef typename PrecomputedWarpPreconditionerInternalData<TDataTypes>::TBaseVector TVector;
    typedef typename PrecomputedWarpPreconditionerInternalData<TDataTypes>::TBaseMatrix TBaseMatrix;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(PrecomputedWarpPreconditioner,TDataTypes),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));
    typedef TDataTypes DataTypes;
    typedef typename TDataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename TDataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename behavior::MechanicalState<DataTypes> MState;

    typedef MatNoInit<3, 3, Real> Transformation;

    Data<bool> jmjt_twostep;
    Data<bool> f_verbose;
    Data<bool> use_file;
    Data<bool> share_matrix;
    Data <std::string> solverName;
    Data<bool> use_rotations;
    Data<double> draw_rotations_scale;
    Data<helper::vector<int> > sub_compliance_index;

    MState * mstate;

    PrecomputedWarpPreconditioner();

    void solve (TMatrix& M, TVector& x, TVector& b);
    void invert(TMatrix& M);
    void setSystemMBKMatrix(const core::MechanicalParams* mparams);
    bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact);
    void draw();
    void init();
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
        if (dynamic_cast<MState *>(context->getMechanicalState()) == NULL) return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const PrecomputedWarpPreconditioner<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected :
    TVector R;
    TVector T;
    TVector subR;
    TVector subZ;
    PrecomputedWarpPreconditionerInternalData<TDataTypes> internalData;

    void rotateConstraints();
    void loadMatrixWithCSparse(TMatrix& M);
    void loadMatrixWithSolver();

    template<class JMatrix>
    void ComputeResult(defaulttype::BaseMatrix * result,JMatrix& J, float fact);

    template<class JMatrix>
    void filterSubJ(JMatrix& J);

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
    SparseMatrix<Real> subJ;
    std::vector<int> sub_sorted_index;
};


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
