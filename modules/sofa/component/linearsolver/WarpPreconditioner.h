/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_WARPPRECONDITIONER_H
#define SOFA_COMPONENT_LINEARSOLVER_WARPPRECONDITIONER_H

#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/linearsolver/FullVector.h>
#include <math.h>
#include <sofa/core/behavior/RotationMatrix.h>
#include <sofa/core/behavior/BaseRotationFinder.h>
#include <sofa/core/behavior/RotationMatrix.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>

#include <map>

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class DataTypes>
class WarpPreconditionerInternalData
{
public:
    typedef typename DataTypes::Real Real;
    typedef RotationMatrix<Real> TRotationMatrix;
    typedef FullVector<Real> TVector;
    typedef typename sofa::component::linearsolver::SparseMatrix<Real> JMatrixType;

    template<typename MReal>
    JMatrixType * copyJmatrix(SparseMatrix<MReal> * J)
    {
        J_local.clear();
        J_local.resize(J->rowSize(),J->colSize());

        for (typename sofa::component::linearsolver::SparseMatrix<MReal>::LineConstIterator jit1 = J->begin(); jit1 != J->end(); jit1++)
        {
            int l = jit1->first;
            for (typename sofa::component::linearsolver::SparseMatrix<MReal>::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
            {
                int c = i1->first;
                MReal val = i1->second;
                J_local.set(l,c,val);
            }
        }
        return &J_local;
    }

    JMatrixType * getLocalJ(defaulttype::BaseMatrix* J)
    {
        if (JMatrixType * j = dynamic_cast<JMatrixType *>(J))
        {
            return j;
        }
        else if (SparseMatrix<double> * j = dynamic_cast<SparseMatrix<double> *>(J))
        {
            return copyJmatrix(j);
        }
        else if (SparseMatrix<float> * j = dynamic_cast<SparseMatrix<float> *>(J))
        {
            return copyJmatrix(j);
        }
        else
        {
            J_local.clear();
            J_local.resize(J->rowSize(),J->colSize());

            for (unsigned j=0; j<J->rowSize(); j++)
            {
                for (unsigned i=0; i<J->colSize(); i++)
                {
                    J_local.set(j,i,J->element(j,i));
                }
            }

            return &J_local;
        }
    }

    void opMulJ(TRotationMatrix * R,JMatrixType * J)
    {
        for (typename sofa::component::linearsolver::SparseMatrix<Real>::LineConstIterator jit1 = J->begin(); jit1 != J->end(); jit1++)
        {
            int l = jit1->first;
            for (typename sofa::component::linearsolver::SparseMatrix<Real>::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end();)
            {
                int c = i1->first;
                Real v0 = (Real)i1->second; i1++; if (i1==jit1->second.end()) break;
                Real v1 = (Real)i1->second; i1++; if (i1==jit1->second.end()) break;
                Real v2 = (Real)i1->second; i1++;
                J->set(l,c+0,v0 * R->getVector()[(c+0)*3+0] + v1 * R->getVector()[(c+1)*3+0] + v2 * R->getVector()[(c+2)*3+0] );
                J->set(l,c+1,v0 * R->getVector()[(c+0)*3+1] + v1 * R->getVector()[(c+1)*3+1] + v2 * R->getVector()[(c+2)*3+1] );
                J->set(l,c+2,v0 * R->getVector()[(c+0)*3+2] + v1 * R->getVector()[(c+1)*3+2] + v2 * R->getVector()[(c+2)*3+2] );
            }
        }
    }

private :
    JMatrixType J_local;
};

/// Linear system solver wrapping another (precomputed) linear solver by a per-node rotation matrix
template<class DataTypes>
class WarpPreconditioner : public core::behavior::LinearSolver
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(WarpPreconditioner,DataTypes),sofa::core::behavior::LinearSolver);
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef sofa::defaulttype::MatNoInit<3, 3, Real> Transformation;
    typedef sofa::core::behavior::LinearSolver Inherit;

    typedef typename WarpPreconditionerInternalData<DataTypes>::TRotationMatrix TRotationMatrix;
    typedef typename WarpPreconditionerInternalData<DataTypes>::TVector TVector;
    typedef typename WarpPreconditionerInternalData<DataTypes>::JMatrixType JMatrixType;

    Data<bool> f_verbose;
    Data <std::string> solverName;
    Data<unsigned> f_useRotationFinder;
    Data<bool> f_enable;
    Data<double> f_draw_rotations_scale;

protected:
    WarpPreconditioner();

public:

    ~WarpPreconditioner();

    void bwdInit();

    virtual void setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams);

    virtual void solveSystem();

    virtual bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact);

    unsigned getSystemDimention();

    virtual void resetSystem();

    virtual void invertSystem();

    virtual void setSystemLHVector(core::MultiVecDerivId v);

    virtual void setSystemRHVector(core::MultiVecDerivId v);

    virtual bool addMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact);

    virtual defaulttype::BaseMatrix* getSystemBaseMatrix();

    virtual defaulttype::BaseVector* getSystemRHBaseVector();

    virtual defaulttype::BaseVector* getSystemLHBaseVector();

    virtual defaulttype::BaseMatrix* getSystemInverseBaseMatrix();

    virtual bool readFile(std::istream& in);

    virtual bool writeFile(std::ostream& out);

    virtual void freezeSystemMatrix();

    virtual void updateSystemMatrix();

    virtual void draw(const core::visual::VisualParams* vparams);

    virtual bool isAsyncSolver();

    TRotationMatrix* createRotationMatrix()
    {
        return new TRotationMatrix;
    }

    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const WarpPreconditioner<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    template<class T>
    void executeVisitor(T v)
    {
        v.setTags(this->getTags());
        v.execute( this->getContext() );
    }


private :

    core::behavior::LinearSolver* realSolver;
    core::behavior::MechanicalState<DataTypes>* mstate;

    core::MultiVecDerivId systemLHVId,systemRHVId;
    GraphScatteredVector * tmpVecId;

    TVector tmpVector1,tmpVector2;

    int indRotationFinder;
    core::MechanicalParams params;

    int updateSystemSize,currentSystemSize;

    int indexwork;
    bool first;

    TRotationMatrix Rcur;
    TRotationMatrix * rotationWork[2];
    std::vector<sofa::core::behavior::BaseRotationFinder *> rotationFinders;
    WarpPreconditionerInternalData<DataTypes> internalData;
};


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
