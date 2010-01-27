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
#ifndef SOFA_COMPONENT_LINEARSOLVER_PCGLinearSolver_H
#define SOFA_COMPONENT_LINEARSOLVER_PCGLinearSolver_H

#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/helper/map.h>

//#define DISPLAY_TIME 200

#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

#ifdef DISPLAY_TIME
#include <sofa/helper/system/thread/CTime.h>
using sofa::helper::system::thread::CTime;
#endif


/// Linear system solver using the conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
class PCGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(PCGLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;
    typedef sofa::core::componentmodel::behavior::BaseMechanicalState::VecId VecId;
    Data<unsigned> f_maxIter;
    Data<double> f_tolerance;
    Data<double> f_smallDenominatorThreshold;
    Data<bool> f_verbose;
    Data<int> f_refresh;
    Data<bool> use_precond;
    Data< helper::vector< std::string > > f_preconditioners;
#ifdef DISPLAY_TIME
    Data<bool> display_time;
#endif
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graph;
    std::vector<sofa::core::componentmodel::behavior::LinearSolver*> preconditioners;

    PCGLinearSolver()
        : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
        , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
        , f_smallDenominatorThreshold( initData(&f_smallDenominatorThreshold,1e-5,"threshold","minimum value of the denominator in the conjugate Gradient solution") )
        , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
        , f_refresh( initData(&f_refresh,0,"refresh","Refresh iterations") )
        , use_precond( initData(&use_precond,true,"precond","Use preconditioners") )
        , f_preconditioners( initData(&f_preconditioners, "preconditioners", "If not empty: path to the solvers to use as preconditioners") )
#ifdef DISPLAY_TIME
        , display_time( initData(&display_time,false,"display_time","display time information") )
#endif
        , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
    {
        f_graph.setWidget("graph");
        f_graph.setReadOnly(true);
        iteration = 0;
        no_precond = false;
#ifdef DISPLAY_TIME
        timeStamp = 1.0 / (double)CTime::getRefTicksPerSec();
#endif
    }

    void solve (Matrix& M, Vector& x, Vector& b);
    void init();
    void setSystemMBKMatrix(double mFact=0.0, double bFact=0.0, double kFact=0.0);
    //void setSystemRHVector(VecId v);
    //void setSystemLHVector(VecId v);

private :
    int iteration;
    bool no_precond;
#ifdef DISPLAY_TIME
    double time1;
    double time2;
    double time3;
    double time4;
    double timeStamp;
    int step_simu;
    int it_simu;
#endif
protected:
    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: p = p*beta + r
    inline void cgstep_beta(Vector& p, Vector& r, double beta);
    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: x += p*alpha, r -= q*alpha
    inline void cgstep_alpha(Vector& x, Vector& r, Vector& p, Vector& q, double alpha);
};

template<class TMatrix, class TVector>
inline void PCGLinearSolver<TMatrix,TVector>::cgstep_beta(Vector& p, Vector& r, double beta)
{
    p *= beta;
    p += r; //z;
}

template<class TMatrix, class TVector>
inline void PCGLinearSolver<TMatrix,TVector>::cgstep_alpha(Vector& x, Vector& r, Vector& p, Vector& q, double alpha)
{
    x.peq(p,alpha);                 // x = x + alpha p
    r.peq(q,-alpha);                // r = r - alpha q
}

template<>
inline void PCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(Vector& p, Vector& r, double beta)
{
    this->v_op(p,r,p,beta); // p = p*beta + r
}

template<>
inline void PCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(Vector& x, Vector& r, Vector& p, Vector& q, double alpha)
{
#if 1 //SOFA_NO_VMULTIOP // unoptimized version
    x.peq(p,alpha);                 // x = x + alpha p
    r.peq(q,-alpha);                // r = r - alpha q
#else // single-operation optimization
    typedef core::componentmodel::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp ops;
    ops.resize(2);
    ops[0].first = (VecId)x;
    ops[0].second.push_back(std::make_pair((VecId)x,1.0));
    ops[0].second.push_back(std::make_pair((VecId)p,alpha));
    ops[1].first = (VecId)r;
    ops[1].second.push_back(std::make_pair((VecId)r,1.0));
    ops[1].second.push_back(std::make_pair((VecId)q,-alpha));
    simulation::tree::MechanicalVMultiOpVisitor vmop(ops);
    vmop.execute(this->getContext());
#endif
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
