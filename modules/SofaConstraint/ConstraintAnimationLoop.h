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
#ifndef SOFA_COMPONENT_ANIMATIONLOOP_CONSTRAINTANIMATIONLOOP_H
#define SOFA_COMPONENT_ANIMATIONLOOP_CONSTRAINTANIMATIONLOOP_H

#include <sofa/SofaGeneral.h>

#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/OdeSolver.h>

#include <SofaBaseLinearSolver/FullMatrix.h>

#include <sofa/simulation/common/CollisionAnimationLoop.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/Node.h>


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
    MechanicalGetConstraintResolutionVisitor(const core::ConstraintParams* params /* PARAMS FIRST */, std::vector<core::behavior::ConstraintResolution*>& res, unsigned int offset)
        : simulation::BaseMechanicalVisitor(params), _cparams(params), _res(res),_offset(offset)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
        //serr<<"creation of the visitor"<<sendl;
    }

    virtual Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet)
    {
        //serr<<"fwdConstraint called on "<<c->getName()<<sendl;

        if (core::behavior::BaseConstraint *c=dynamic_cast<core::behavior::BaseConstraint*>(cSet))
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
    MechanicalSetConstraint(const core::ConstraintParams* _cparams /* PARAMS FIRST  = sofa::core::ConstraintParams::defaultInstance()*/, core::MultiMatrixDerivId _res, unsigned int &_contactId)
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

        c->buildConstraintMatrix(cparams /* PARAMS FIRST */, res, contactId);

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
    MechanicalAccumulateConstraint2(const core::ConstraintParams* _cparams /* PARAMS FIRST  = sofa::core::ConstraintParams::defaultInstance()*/, core::MultiMatrixDerivId _res)
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
        map->applyJT(cparams /* PARAMS FIRST */, res, res);
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

    inline int getSize(void) {return _dim;};
    inline sofa::component::linearsolver::LPtrFullMatrix<double>* getW(void) {return &_W;};
    inline sofa::component::linearsolver::FullVector<double>* getDfree(void) {return &_dFree;};
    inline sofa::component::linearsolver::FullVector<double>* getD(void) {return &_d;};
    inline sofa::component::linearsolver::FullVector<double>* getF(void) {return &_force;};
    inline sofa::component::linearsolver::FullVector<double>* getdF(void) {return &_df;};
    inline std::vector<core::behavior::ConstraintResolution*>& getConstraintResolutions(void) {return _constraintsResolutions;};
    inline double *getTolerance(void) {return &_tol;};

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
    // virtual const char* getTypeName() const { return "AnimationLoop"; }

    virtual void step(const core::ExecParams* params /* PARAMS FIRST */, double dt);

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

    ConstraintProblem *getConstraintProblem(void) {return (bufCP1 == true) ? &CP1 : &CP2;};

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
    void freeMotion(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context, double &dt);
    void setConstraintEquations(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context);
    void correctiveMotion(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context);
    void debugWithContact(int numConstraints);

    ///  Specific procedures that are called for setting the constraints:

    /// 1.calling resetConstraint & setConstraint & accumulateConstraint visitors
    /// and resize the constraint problem that will be solved
    void writeAndAccumulateAndCountConstraintDirections(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context, unsigned int &numConstraints);

    /// 2.calling GetConstraintViolationVisitor: each constraint provides its present violation
    /// for a given state (by default: free_position TODO: add VecId to make this method more generic)
    void getIndividualConstraintViolations(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context);

    /// 3.calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
    void getIndividualConstraintSolvingProcess(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context);

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
    double timeScale, time ;
    bool debug;

    unsigned int numConstraints;

    bool bufCP1;
    double compTime, iterationTime;

private:
    ConstraintProblem CP1, CP2;
};

} // namespace animationloop

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_ANIMATIONLOOP_CONSTRAINTANIMATIONLOOP_H */
