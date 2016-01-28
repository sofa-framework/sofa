/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <SofaPreconditioner/ShewchukPCGLinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaDenseSolver/NewMatMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/AdvancedTimer.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
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
using std::cerr;
using std::endl;

template<class TMatrix, class TVector>
ShewchukPCGLinearSolver<TMatrix,TVector>::ShewchukPCGLinearSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_use_precond( initData(&f_use_precond,true,"use_precond","Use preconditioner") )
    , f_update_step( initData(&f_update_step,(unsigned)1,"update_step","Number of steps before the next refresh of precondtioners") )
    , f_build_precond( initData(&f_build_precond,true,"build_precond","Build the preconditioners, if false build the preconditioner only at the initial step") )
    , f_preconditioners( initData(&f_preconditioners, "preconditioners", "If not empty: path to the solvers to use as preconditioners") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )    
    , f_target(initData(&f_target, "target", "target to the node we want to solve"))
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

    if (first)  //We initialize all the preconditioners for the first step
    {
        preconditioners->setSystemMBKMatrix(mparams);
        first = false;
        next_refresh_step = 1;
    }
    else if (f_build_precond.getValue())
    {
        sofa::helper::AdvancedTimer::valSet("PCG::PrecondBuildMBK", 1);
        sofa::helper::AdvancedTimer::stepBegin("PCG::PrecondSetSystemMBKMatrix");

        if (f_update_step.getValue()>0)
        {
            if (next_refresh_step>=f_update_step.getValue())
            {
                preconditioners->setSystemMBKMatrix(mparams);
                next_refresh_step=1;
            }
            else
            {
                next_refresh_step++;
            }
        }

        sofa::helper::AdvancedTimer::stepEnd("PCG::PrecondSetSystemMBKMatrix");
    }

    preconditioners->updateSystemMatrix();
}

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(const core::ExecParams* params, Vector& p, Vector& r, double beta)
{
    //std::cout<<"ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta is called !!!!!!!!!!!!"<<std::endl;
    //p.eq(r,p,beta); // p = p*beta + r
    this->executeVisitor( MechanicalVOpVisitor(params, p, r, p, beta) );
//    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>* mstate;
//    this->getContext()->get(mstate, f_target.getValue());
//    MechanicalVOpVisitor v (params, p, r, p, beta);
//    v.execute(mstate->getContext());

}

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(const core::ExecParams* params, Vector& x, Vector& r, Vector& p, Vector& q, SReal alpha)
{
    //std::cout<<"ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha is called !!!!!!!!!!!!"<<std::endl;
    //x.peq(p,alpha);                 // x = x + alpha p
    //r.peq(q,-alpha);

#ifdef SOFA_NO_VMULTIOP // unoptimized version
    x.peq(p,alpha);                 // x = x + alpha p
    r.peq(q,-alpha);                // r = r - alpha q
#else // single-operation optimization
    typedef sofa::core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp ops;
    ops.resize(2);
    ops[0].first = (sofa::core::MultiVecDerivId)x;
    ops[0].second.push_back(std::make_pair((sofa::core::MultiVecDerivId)x,1.0));
    ops[0].second.push_back(std::make_pair((sofa::core::MultiVecDerivId)p,alpha));
    ops[1].first = (sofa::core::MultiVecDerivId)r;
    ops[1].second.push_back(std::make_pair((sofa::core::MultiVecDerivId)r,1.0));
    ops[1].second.push_back(std::make_pair((sofa::core::MultiVecDerivId)q,-alpha));
    this->executeVisitor(simulation::MechanicalVMultiOpVisitor(params, ops));
//    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>* mstate;
//    this->getContext()->get(mstate, f_target.getValue());
//    simulation::MechanicalVMultiOpVisitor v (params, ops);
//    v.execute(mstate->getContext());
#endif
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
void ShewchukPCGLinearSolver<TMatrix,TVector>::solve (Matrix& M, Vector& x, Vector& b)// M here is the matrix of the system
{
    sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");

    std::map < std::string, sofa::helper::vector<double> >& graph = * f_graph.beginEdit();
//    sofa::helper::vector<double>& graph_error = graph["Error"];

    const char* endcond = "iterations";

    newton_iter++;
    char name[256];
    sprintf(name,"Error %d",newton_iter);
    sofa::helper::vector<double>& graph_error = graph[std::string(name)];

    const core::ExecParams* params = core::ExecParams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    //std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve, this->currentNode->getName() = "<< this->currentNode->getName() <<std::endl;
    Vector& r = *vtmp.createTempVector();
    Vector& w = *vtmp.createTempVector();
    Vector& s = *vtmp.createTempVector();

    bool apply_precond = preconditioners!=NULL && f_use_precond.getValue();

    //std::cout<<std::endl<<std::endl;
    //std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve,  apply_precond = "<<apply_precond<<std::endl;

    double b_norm = b.dot(b);
    double tol = f_tolerance.getValue() * b_norm;

    r = M * x;

    //std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve, BEFORE r= "<<r<<std::endl;


    cgstep_beta(params,r,b,-1);// r = -1 * r + b  =   b - (M * x): residu

    if (apply_precond)
    {
        //std::cout<<std::endl<<std::endl;
        //std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve, r= "<<r<<std::endl;
        sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");
        preconditioners->setSystemLHVector(w);// w is unknown
        //std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve, r= "<<r.getName() <<std::endl;
//        for(unsigned i=0; i<r.size(); ++i)
//        {
//            std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve, r= "<<r.element(i) <<std::endl;
//        }
        preconditioners->setSystemRHVector(r);
        preconditioners->solveSystem();/// solve Precond*w=r to find w = Precond^-1 * r
        //std::cout<<std::endl<<std::endl;
        //std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve, w= "<<w<<std::endl;


        sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
    }
    else
    {
        w = r;// w: search direction
    }

    double r_norm = r.dot(w);
    graph_error.push_back(r_norm/b_norm);

    //std::cout<<std::endl<<std::endl;
    //std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve,  r_norm = "<<r_norm<<std::endl;

    if(r_norm<=tol)
    {
        endcond = "tolerance";
    }

    const bool printLog = this->f_printLog.getValue();

    unsigned iter=1;
    while ((iter <= f_maxIter.getValue()) && (r_norm > tol))
    {
        s = M * w;
        double dtq = w.dot(s);
        double alpha = r_norm / dtq;

        //cgstep_alpha(x,w,alpha);//for(int i=0; i<n; i++) x[i] += alpha * d[i];
        //cgstep_alpha(r,s,-alpha);//for (int i=0; i<n; i++) r[i] = r[i] - alpha * q[i];
        cgstep_alpha(params,x, r,w,s, alpha);// x=x+alpha*w and r=r-alpha*s


        if (apply_precond)
        {
            //std::cout<<std::endl<<std::endl;
            //std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve, r= "<<r<<std::endl;
            sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");
            preconditioners->setSystemLHVector(s);// s is unknown
            preconditioners->setSystemRHVector(r);
            preconditioners->solveSystem();// solve Precond*s = r to find s = Precond^-1*r

            //std::cout<<"ShewchukPCGLinearSolver<TMatrix,TVector>::solve, s= "<<s<<std::endl;

            sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
        } else
        {
            s = r;
        }

        double r_norm_old = r_norm;
        r_norm = r.dot(s);
        if(r_norm<=tol)
        {
            endcond = "tolerance";
        }
        graph_error.push_back(r_norm/b_norm);

        double beta = r_norm / r_norm_old;

        //for (int i=0; i<n; i++) d[i] = r[i] + beta * d[i];
        cgstep_beta(params,w,s,beta);//w=w*beta+s

        iter++;
    }

    if( printLog )
    {
        std::cout<<"ShewchukPCGLinearSolver::solve, nbiter = "<<iter<<" stop because of "<<endcond<<std::endl;
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
//.add< ShewchukPCGLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> > >()
.addAlias("PCGLinearSolver");
;

} // namespace linearsolver

} // namespace component

} // namespace sofa

