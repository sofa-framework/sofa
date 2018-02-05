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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <SofaPreconditioner/ShewchukPCGLinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/AdvancedTimer.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

template<class TMatrix, class TVector>
ShewchukPCGLinearSolver<TMatrix,TVector>::ShewchukPCGLinearSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_use_precond( initData(&f_use_precond,true,"use_precond","Use preconditioner") )
    , f_update_step( initData(&f_update_step,(unsigned)1,"update_step","Number of steps before the next refresh of precondtioners") )
    , f_build_precond( initData(&f_build_precond,true,"build_precond","Build the preconditioners, if false build the preconditioner only at the initial step") )
    , f_preconditioners( initData(&f_preconditioners, "preconditioners", "If not empty: path to the solvers to use as preconditioners") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
{
    f_graph.setWidget("graph");
//    f_graph.setReadOnly(true);
    first = true;
    this->f_listening.setValue(true);
}

template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::init()
{
    if (! f_preconditioners.getValue().empty()) {
        BaseContext * c = this->getContext();

        c->get(preconditioners, f_preconditioners.getValue());

        if (preconditioners) sout << "Found " << f_preconditioners.getValue() << sendl;
        else serr << "Solver \"" << f_preconditioners.getValue() << "\" not found." << sendl;
    }

    first = true;
}

template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    sofa::helper::AdvancedTimer::valSet("PCG::buildMBK", 1);
    sofa::helper::AdvancedTimer::stepBegin("PCG::setSystemMBKMatrix");

    Inherit::setSystemMBKMatrix(mparams);

    sofa::helper::AdvancedTimer::stepEnd("PCG::setSystemMBKMatrix");

    if (preconditioners==NULL) return;

    if (first) {  //We initialize all the preconditioners for the first step
        preconditioners->setSystemMBKMatrix(mparams);
        first = false;
        next_refresh_step = 1;

    } else if (f_build_precond.getValue()) {
        sofa::helper::AdvancedTimer::valSet("PCG::PrecondBuildMBK", 1);
        sofa::helper::AdvancedTimer::stepBegin("PCG::PrecondSetSystemMBKMatrix");

        if (f_update_step.getValue()>0) {
            if (next_refresh_step>=f_update_step.getValue()) {
                preconditioners->setSystemMBKMatrix(mparams);
                next_refresh_step=1;
            } else {
                next_refresh_step++;
            }
        }

        sofa::helper::AdvancedTimer::stepEnd("PCG::PrecondSetSystemMBKMatrix");
    }

    preconditioners->updateSystemMatrix();
}

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(Vector& p, Vector& r, double beta)
{
    p.eq(r,p,beta); // p = p*beta + r
}

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(Vector& x, Vector& p, double alpha)
{
    x.peq(p,alpha);                 // x = x + alpha p
}

template<class Matrix, class Vector>
void ShewchukPCGLinearSolver<Matrix,Vector>::handleEvent(sofa::core::objectmodel::Event* event) {
    /// this event shoul be launch before the addKToMatrix
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event)) {
        newton_iter = 0;
        std::map < std::string, sofa::helper::vector<double> >& graph = * f_graph.beginEdit();
        graph.clear();
    }
}


template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::solve (Matrix& M, Vector& x, Vector& b)
{
    sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");

    std::map < std::string, sofa::helper::vector<double> >& graph = * f_graph.beginEdit();
//    sofa::helper::vector<double>& graph_error = graph["Error"];

    newton_iter++;
    char name[256];
    sprintf(name,"Error %d",newton_iter);
    sofa::helper::vector<double>& graph_error = graph[std::string(name)];

    const core::ExecParams* params = core::ExecParams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    Vector& r = *vtmp.createTempVector();
    Vector& w = *vtmp.createTempVector();
    Vector& s = *vtmp.createTempVector();

    bool apply_precond = preconditioners!=NULL && f_use_precond.getValue();

    double b_norm = b.dot(b);
    double tol = f_tolerance.getValue() * b_norm;

    r = M * x;
    cgstep_beta(r,b,-1);// r = -1 * r + b  =   b - (M * x)

    if (apply_precond) {
        sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");
        preconditioners->setSystemLHVector(w);
        preconditioners->setSystemRHVector(r);
        preconditioners->solveSystem();

        sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
    } else {
        w = r;
    }

    double r_norm = r.dot(w);
    graph_error.push_back(r_norm/b_norm);

    unsigned iter=1;
    while ((iter <= f_maxIter.getValue()) && (r_norm > tol)) {
        s = M * w;
        double dtq = w.dot(s);
        double alpha = r_norm / dtq;

        cgstep_alpha(x,w,alpha);//for(int i=0; i<n; i++) x[i] += alpha * d[i];
        cgstep_alpha(r,s,-alpha);//for (int i=0; i<n; i++) r[i] = r[i] - alpha * q[i];

        if (apply_precond) {
            sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");
            preconditioners->setSystemLHVector(s);
            preconditioners->setSystemRHVector(r);
            preconditioners->solveSystem();

            sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
        } else {
            s = r;
        }

        double deltaOld = r_norm;
        r_norm = r.dot(s);
        graph_error.push_back(r_norm/b_norm);

        double beta = r_norm / deltaOld;

        cgstep_beta(w,s,beta);//for (int i=0; i<n; i++) d[i] = r[i] + beta * d[i];

        iter++;
    }

    f_graph.endEdit();

    vtmp.deleteTempVector(&r);
    vtmp.deleteTempVector(&w);
    vtmp.deleteTempVector(&s);

    sofa::helper::AdvancedTimer::valSet("PCG iterations", iter);
    sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
}

SOFA_DECL_CLASS(ShewchukPCGLinearSolver)

int ShewchukPCGLinearSolverClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
.add< ShewchukPCGLinearSolver<GraphScatteredMatrix,GraphScatteredVector> >(true)
.addAlias("PCGLinearSolver");
;

} // namespace linearsolver

} // namespace component

} // namespace sofa

