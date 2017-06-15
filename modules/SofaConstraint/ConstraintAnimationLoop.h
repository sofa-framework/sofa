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
#ifndef SOFA_COMPONENT_ANIMATIONLOOP_CONSTRAINTANIMATIONLOOP_H
#define SOFA_COMPONENT_ANIMATIONLOOP_CONSTRAINTANIMATIONLOOP_H
#include "config.h"

#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/OdeSolver.h>

#include <SofaBaseLinearSolver/FullMatrix.h>

#include <sofa/simulation/CollisionAnimationLoop.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/Node.h>


#include <vector>

namespace sofa
{

namespace component
{

namespace animationloop
{


class SOFA_CONSTRAINT_API MechanicalGetConstraintResolutionVisitor : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalGetConstraintResolutionVisitor(const core::ConstraintParams* params, std::vector<core::behavior::ConstraintResolution*>& res, unsigned int offset)
        : simulation::BaseMechanicalVisitor(params), _res(res),_offset(offset), _cparams(params)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
        //serr<<"creation of the visitor"<<sendl;
    }

    virtual Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet)
    {
        //serr<<"fwdConstraint called on "<<c->getName()<<sendl;

        if (core::behavior::BaseConstraint *c=cSet->toBaseConstraint())
        {
            ctime_t t0 = begin(node, c);
            c->getConstraintResolution(_cparams, _res, _offset);
            end(node, c, t0);
        }
        return RESULT_CONTINUE;
    }

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
private:
    std::vector<core::behavior::ConstraintResolution*>& _res;
    unsigned int _offset;
    const sofa::core::ConstraintParams *_cparams;
};


class SOFA_CONSTRAINT_API MechanicalSetConstraint : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalSetConstraint(const core::ConstraintParams* _cparams, core::MultiMatrixDerivId _res, unsigned int &_contactId)
        : simulation::BaseMechanicalVisitor(_cparams)
        , res(_res)
        , contactId(_contactId)
        , cparams(_cparams)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* c)
    {
        ctime_t t0 = begin(node, c);

        c->setConstraintId(contactId);
        c->buildConstraintMatrix(cparams, res, contactId);

        end(node, c, t0);
        return RESULT_CONTINUE;
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const
    {
        return "MechanicalSetConstraint";
    }

    virtual bool isThreadSafe() const
    {
        return false;
    }

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

protected:

    sofa::core::MultiMatrixDerivId res;
    unsigned int &contactId;
    const sofa::core::ConstraintParams *cparams;
};


class SOFA_CONSTRAINT_API MechanicalAccumulateConstraint2 : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalAccumulateConstraint2(const core::ConstraintParams* _cparams, core::MultiMatrixDerivId _res)
        : simulation::BaseMechanicalVisitor(_cparams)
        , res(_res)
        , cparams(_cparams)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual void bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map)
    {
        ctime_t t0 = begin(node, map);
        map->applyJT(cparams, res, res);
        end(node, map, t0);
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAccumulateConstraint2"; }

    virtual bool isThreadSafe() const
    {
        return false;
    }
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

protected:
    core::MultiMatrixDerivId res;
    const sofa::core::ConstraintParams *cparams;
};


class SOFA_CONSTRAINT_API ConstraintProblem
{
protected:
    sofa::component::linearsolver::LPtrFullMatrix<double> _W;
    sofa::component::linearsolver::FullVector<double> _dFree, _force, _d, _df;              // cf. These Duriez + _df for scheme correction
    std::vector<core::behavior::ConstraintResolution*> _constraintsResolutions;
    double _tol;
    int _dim;
    sofa::helper::system::thread::CTime *_timer;
    bool m_printLog;

public:
    ConstraintProblem(bool printLog=false);
    virtual ~ConstraintProblem();
    virtual void clear(int dim, const double &tol);

    inline int getSize(void) {return _dim;}
    inline sofa::component::linearsolver::LPtrFullMatrix<double>* getW(void) {return &_W;}
    inline sofa::component::linearsolver::FullVector<double>* getDfree(void) {return &_dFree;}
    inline sofa::component::linearsolver::FullVector<double>* getD(void) {return &_d;}
    inline sofa::component::linearsolver::FullVector<double>* getF(void) {return &_force;}
    inline sofa::component::linearsolver::FullVector<double>* getdF(void) {return &_df;}
    inline std::vector<core::behavior::ConstraintResolution*>& getConstraintResolutions(void) {return _constraintsResolutions;}
    inline double *getTolerance(void) {return &_tol;}

    void gaussSeidelConstraintTimed(double &timeout, int numItMax);
};




class SOFA_CONSTRAINT_API ConstraintAnimationLoop : public sofa::simulation::CollisionAnimationLoop
{
public:
    typedef sofa::simulation::CollisionAnimationLoop Inherit;

    SOFA_CLASS(ConstraintAnimationLoop, sofa::simulation::CollisionAnimationLoop);
protected:
    ConstraintAnimationLoop(simulation::Node* gnode = NULL);
    virtual ~ConstraintAnimationLoop();
public:

    virtual void step(const core::ExecParams* params, SReal dt);

    //virtual void propagatePositionAndVelocity(double t, VecId x, VecId v);

    virtual void init();

    Data<bool> displayTime;
    Data<double> _tol;
    Data<int> _maxIt;
    Data<bool> doCollisionsFirst;
    Data<bool> doubleBuffer;
    Data<bool> scaleTolerance;
    Data<bool> _allVerified;
    Data<double> _sor;
    Data<bool> schemeCorrection;
    Data<bool> _realTimeCompensation;

    Data<bool> activateSubGraph;

    Data<std::map < std::string, sofa::helper::vector<double> > > _graphErrors, _graphConstraints, _graphForces;

    ConstraintProblem *getConstraintProblem(void) {return (bufCP1 == true) ? &CP1 : &CP2;}

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T*, BaseContext* context, BaseObjectDescription* arg)
    {
        simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
        typename T::SPtr obj = sofa::core::objectmodel::New<T>(gnode);
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
        return obj;
    }

protected:
    void launchCollisionDetection(const core::ExecParams* params);
    void freeMotion(const core::ExecParams* params, simulation::Node *context, SReal &dt);
    void setConstraintEquations(const core::ExecParams* params, simulation::Node *context);
    void correctiveMotion(const core::ExecParams* params, simulation::Node *context);
    void debugWithContact(int numConstraints);

    ///  Specific procedures that are called for setting the constraints:

    /// 1.calling resetConstraint & setConstraint & accumulateConstraint visitors
    /// and resize the constraint problem that will be solved
    void writeAndAccumulateAndCountConstraintDirections(const core::ExecParams* params, simulation::Node *context, unsigned int &numConstraints);

    /// 2.calling GetConstraintViolationVisitor: each constraint provides its present violation
    /// for a given state (by default: free_position TODO: add VecId to make this method more generic)
    void getIndividualConstraintViolations(const core::ExecParams* params, simulation::Node *context);

    /// 3.calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
    void getIndividualConstraintSolvingProcess(const core::ExecParams* params, simulation::Node *context);

    /// 4.calling addComplianceInConstraintSpace projected in the contact space => getDelassusOperator(_W) = H*C*Ht
    virtual void computeComplianceInConstraintSpace();


    /// method for predictive scheme:
    void computePredictiveForce(int dim, double* force, std::vector<core::behavior::ConstraintResolution*>& res);



    void gaussSeidelConstraint(int dim, double* dfree, double** w, double* force, double* d, std::vector<core::behavior::ConstraintResolution*>& res, double* df);

    std::vector<core::behavior::BaseConstraintCorrection*> constraintCorrections;


    virtual ConstraintProblem *getCP()
    {
        if (doubleBuffer.getValue() && bufCP1)
            return &CP2;
        else
            return &CP1;
    }

    sofa::helper::system::thread::CTime *timer;
    SReal timeScale, time ;


    unsigned int numConstraints;

    bool bufCP1;
    SReal compTime, iterationTime;

private:
    ConstraintProblem CP1, CP2;
};

} // namespace animationloop

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_ANIMATIONLOOP_CONSTRAINTANIMATIONLOOP_H */
