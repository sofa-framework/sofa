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
#ifndef SOFA_COMPONENT_MASTERSOLVER_MASTERCONSTRAINTSOLVER_H
#define SOFA_COMPONENT_MASTERSOLVER_MASTERCONSTRAINTSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/common/MasterSolverImpl.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/component/odesolver/OdeSolverImpl.h>
#include <sofa/component/linearsolver/FullMatrix.h>

#include <vector>

namespace sofa
{

namespace component
{

namespace mastersolver
{

using namespace sofa::defaulttype;
using namespace sofa::component::linearsolver;
using namespace helper::system::thread;

class SOFA_COMPONENT_MASTERSOLVER_API MechanicalGetConstraintResolutionVisitor : public simulation::MechanicalVisitor
{
public:
    MechanicalGetConstraintResolutionVisitor(std::vector<core::componentmodel::behavior::ConstraintResolution*>& res, unsigned int offset = 0)
        : _res(res),_offset(offset)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
        //serr<<"creation of the visitor"<<sendl;
    }

    virtual Result fwdConstraint(simulation::Node* node, core::componentmodel::behavior::BaseConstraint* c)
    {
        //serr<<"fwdConstraint called on "<<c->getName()<<sendl;

        ctime_t t0 = begin(node, c);
        c->getConstraintResolution(_res, _offset);
        end(node, c, t0);
        return RESULT_CONTINUE;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
private:
    std::vector<core::componentmodel::behavior::ConstraintResolution*>& _res;
    unsigned int _offset;
};

class SOFA_COMPONENT_MASTERSOLVER_API MechanicalSetConstraint : public simulation::MechanicalVisitor
{
public:
    MechanicalSetConstraint(unsigned int &_contactId)
        :contactId(_contactId)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdConstraint(simulation::Node* node, core::componentmodel::behavior::BaseConstraint* c)
    {
        ctime_t t0 = begin(node, c);
        c->applyConstraint(contactId);
        end(node, c, t0);
        return RESULT_CONTINUE;
    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalSetConstraint"; }

    virtual bool isThreadSafe() const
    {
        return false;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

protected:
    unsigned int &contactId;
};

class SOFA_COMPONENT_MASTERSOLVER_API MechanicalAccumulateConstraint2 : public simulation::MechanicalVisitor
{
public:
    MechanicalAccumulateConstraint2()
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual void bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
    {
        ctime_t t0 = begin(node, map);
        map->accumulateConstraint();
        end(node, map, t0);
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAccumulateConstraint2"; }

    virtual bool isThreadSafe() const
    {
        return false;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};



class SOFA_COMPONENT_MASTERSOLVER_API ConstraintProblem
{
private:
    LPtrFullMatrix<double> _W;
    FullVector<double> _dFree, _force, _d;              // cf. These Duriez
    std::vector<core::componentmodel::behavior::ConstraintResolution*> _constraintsResolutions;
    double _tol;
    int _dim;
    CTime *_timer;

public:
    ConstraintProblem();
    ~ConstraintProblem();
    void clear(int dim, const double &tol);

    inline int getSize(void) {return _dim;};
    inline LPtrFullMatrix<double>* getW(void) {return &_W;};
    inline FullVector<double>* getDfree(void) {return &_dFree;};
    inline FullVector<double>* getD(void) {return &_d;};
    inline FullVector<double>* getF(void) {return &_force;};
    inline std::vector<core::componentmodel::behavior::ConstraintResolution*>& getConstraintResolutions(void) {return _constraintsResolutions;};
    inline double *getTolerance(void) {return &_tol;};

    void gaussSeidelConstraintTimed(double &timeout, int numItMax);

};
class SOFA_COMPONENT_MASTERSOLVER_API MasterConstraintSolver : public sofa::simulation::MasterSolverImpl
{
public:
    SOFA_CLASS(MasterConstraintSolver, sofa::simulation::MasterSolverImpl);

    MasterConstraintSolver();
    virtual ~MasterConstraintSolver();
    // virtual const char* getTypeName() const { return "MasterSolver"; }

    void step(double dt);

    //virtual void propagatePositionAndVelocity(double t, VecId x, VecId v);

    virtual void init();

    Data<bool> displayTime;
    Data<double> _tol;
    Data<int> _maxIt;
    Data<bool> doCollisionsFirst;
    Data<bool> doubleBuffer;

    ConstraintProblem *getConstraintProblem(void) {return (bufCP1 == true) ? &CP1 : &CP2;};

private:
    void gaussSeidelConstraint(int dim, double* dfree, double** w, double* force, double* d, std::vector<core::componentmodel::behavior::ConstraintResolution*>& res);

    std::vector<core::componentmodel::behavior::BaseConstraintCorrection*> constraintCorrections;


    bool bufCP1;
    ConstraintProblem CP1, CP2;




};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
