
/// CREATING A NEW PYTHON MODULE: _Compliant
///
/// @author Matthieu Nesme
/// @date 2016


#include <SofaPython/PythonMacros.h>
#include <SofaPython/PythonFactory.h>
#include <SofaPython/Binding_Data.h>
#include "Binding_AssembledSystem.h"

#include <sofa/helper/cast.h>
#include <sofa/simulation/Simulation.h>
#include "../assembly/AssemblyVisitor.h"
#include "../odesolver/CompliantImplicitSolver.h"
#include <SofaPython/PythonToSofa.inl>



using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::simulation;
using namespace sofa::component::linearsolver;
using namespace sofa::component::odesolver;
using namespace sofa::core::behavior;



/// args are node + factors m,b,k to return the linear combinaison mM+bB+kK
static PyObject * _Compliant_getAssembledImplicitMatrix(PyObject * /*self*/, PyObject * args)
{
    PyObject* pyNode;

    float M,B,K;
    if (!PyArg_ParseTuple(args, "Offf", &pyNode, &M, &B, &K)) {
        SP_MESSAGE_ERROR( "_Compliant_getAssembledImplicitMatrix: wrong arguments" );
        return NULL;
    }

    BaseNode* node = sofa::py::unwrap<BaseNode>( pyNode );
    if (!node) {
        SP_MESSAGE_ERROR( "_Compliant_getAssembledImplicitMatrix: first argument is not a BaseNode" );
        PyErr_BadArgument();
        return NULL;
    }


//    SP_MESSAGE_INFO( "_Compliant_getAssembledImplicitMatrix: "<<M<<" "<<B<<" "<<K );

    MechanicalParams mparams = *MechanicalParams::defaultInstance();
    mparams.setMFactor( M );
    mparams.setBFactor( B );
    mparams.setKFactor( K );
    AssemblyVisitor assemblyVisitor(&mparams);
    node->getContext()->executeVisitor( &assemblyVisitor );
    AssembledSystem sys;
    assemblyVisitor.assemble(sys); // assemble system


//    SP_MESSAGE_INFO( "_ompliant_getAssembledImplicitMatrix: "<<sys.H );


    // todo returns a sparse matrix

    size_t size = sys.H.rows();

    PyObject* H = PyList_New(size);
    for( size_t row=0 ; row<size ; ++row )
    {
        PyObject* rowpython = PyList_New(size);

        for( size_t col=0 ; col<size ; ++col )
            PyList_SetItem( rowpython, col, PyFloat_FromDouble( sys.H.coeff(row,col) ) );

        PyList_SetItem( H, row, rowpython );
    }


    return H;
}




static PyObject * _Compliant_getImplicitAssembledSystem(PyObject * /*self*/, PyObject * args)
{
    PyObject* pyNode;
    if (!PyArg_ParseTuple(args, "O", &pyNode)) {
        SP_MESSAGE_ERROR( "_Compliant_getAssembledImplicitMatrix: wrong arguments" );
        return NULL;
    }

    sofa::core::objectmodel::BaseNode* node = sofa::py::unwrap<BaseNode>(pyNode);
    if (!node) {
        SP_MESSAGE_ERROR( "_Compliant_getAssembledImplicitMatrix: first argument is not a BaseNode" );
        PyErr_BadArgument();
        return NULL;
    }

    SReal dt = down_cast<Node>(node)->getDt();
    MechanicalParams mparams = *MechanicalParams::defaultInstance();

    // pure implicit coeff, TODO: parametrize these?
    mparams.setMFactor( 1.0 );
    mparams.setBFactor( -dt );
    mparams.setKFactor( -dt*dt );
    mparams.setDt( dt );

    AssemblyVisitor assemblyVisitor(&mparams);
    node->getContext()->executeVisitor( &assemblyVisitor );
    AssembledSystem* sys = new AssembledSystem();
    assemblyVisitor.assemble(*sys); // assemble system

    return SP_BUILD_PYPTR(AssembledSystem,AssembledSystem,sys,true);
}


// takes a CompliantImplicitSolver and a BaseMechanicalState
// returns the lambdas contained in the BaseMechanicalState
// (the corresponding multivecid is in the CompliantImplicitSolver)
// @warning you have to look at the CompliantImplicitSolver's formulation (vel,dv,acc) to deduce constraint forces from lambdas
static PyObject * _Compliant_getLambda(PyObject * /*self*/, PyObject * args)
{
    PyObject* pySolver, *pyState;
    if (!PyArg_ParseTuple(args, "OO", &pySolver, &pyState))
    {
        SP_MESSAGE_ERROR( "_Compliant_getConstraintForce: wrong arguments" );
        return NULL;
    }

    CompliantImplicitSolver* solver = sofa::py::unwrap<CompliantImplicitSolver>( pySolver );
    if (!solver)
    {
        SP_MESSAGE_ERROR( "_Compliant_getConstraintForce: wrong arguments - not a CompliantImplicitSolver" );
        PyErr_BadArgument();
        return NULL;        
    }

    BaseMechanicalState* mstate = sofa::py::unwrap<BaseMechanicalState>( pyState );
    if (!mstate)
    {
        SP_MESSAGE_ERROR( "_Compliant_getConstraintForce: wrong arguments - not a BaseMechanicalState" );
        PyErr_BadArgument();
        return NULL;
    }

    objectmodel::BaseData* data;

    const VecId& vecid = solver->lagrange.id().getId(mstate);
    if( vecid.isNull() )
    {
        SP_MESSAGE_WARNING( "_Compliant_getConstraintForce: allocating lambda vector for mstate "<<mstate->getPathName() )

        VecDerivId id(VecDerivId::V_FIRST_DYNAMIC_INDEX);
        mstate->vAvail( ExecParams::defaultInstance(), id );
        solver->lagrange.id().setId(mstate, id);

        mstate->vAlloc(ExecParams::defaultInstance(),id);

        data = mstate->baseWrite(id);
    } else {
        data = mstate->baseWrite(vecid);
    }
    return SP_BUILD_PYPTR(Data,BaseData,data,false);
}

/// takes a Context and a CompliantImplicitSolver
static PyObject * _Compliant_propagateLambdas(PyObject * /*self*/, PyObject * args)
{
    PyObject* pyNode, *pySolver;
    if (!PyArg_ParseTuple(args, "OO", &pyNode, &pySolver))
    {
        SP_MESSAGE_ERROR( "_Compliant_getConstraintForce: wrong arguments" );
        PyErr_BadArgument();
        return NULL;
    }

    BaseContext* context = sofa::py::unwrap<BaseContext>( pyNode );
    if (!context)
    {
        SP_MESSAGE_ERROR( "_Compliant_getConstraintForce: wrong arguments - not a BaseContext" );
        PyErr_BadArgument();
        return NULL;
    }

    CompliantImplicitSolver* solver = sofa::py::unwrap<CompliantImplicitSolver>( pySolver );
    if (!solver)
    {
        SP_MESSAGE_ERROR( "_Compliant_getConstraintForce: wrong arguments - not a CompliantImplicitSolver" );
        PyErr_BadArgument();
        return NULL;
    }

    propagate_lambdas_visitor vis( MechanicalParams::defaultInstance(), solver->lagrange );
    context->executeVisitor( &vis );

    Py_RETURN_NONE;
}

// Methods of the module
SP_MODULE_METHODS_BEGIN(_Compliant)
SP_MODULE_METHOD(_Compliant,getAssembledImplicitMatrix)
SP_MODULE_METHOD(_Compliant,getImplicitAssembledSystem)
SP_MODULE_METHOD(_Compliant,getLambda)
SP_MODULE_METHOD(_Compliant,propagateLambdas)
SP_MODULE_METHODS_END

