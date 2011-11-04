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
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP
#include <sofa/component/linearsolver/MatrixLinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/MechanicalMatrixVisitor.h>
#include <sofa/simulation/common/MechanicalVPrintVisitor.h>
#include <sofa/simulation/common/VelocityThresholdVisitor.h>
#include <sofa/core/behavior/LinearSolver.h>

#include <stdlib.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using sofa::core::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::resetSystem()
{
//    serr << "resetSystem()" << sendl;
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        if (!currentGroup->systemMatrix)
        {
//            serr << "new systemMatrix" << sendl;
            currentGroup->systemMatrix = new GraphScatteredMatrix();
        }
        if (!currentGroup->systemRHVector)
        {
//            serr << "new systemRHVector" << sendl;
            currentGroup->systemRHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
        }
        if (!currentGroup->systemLHVector)
        {
//            serr << "new systemLHVector" << sendl;
            currentGroup->systemLHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
        }
        currentGroup->systemRHVector->reset();
        currentGroup->systemLHVector->reset();
        currentGroup->solutionVecId = core::MultiVecDerivId::null();
        currentGroup->needInvert = true;
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::resizeSystem(int)
{
//    serr << "resizeSystem()" << sendl;
    if (!currentGroup->systemMatrix) currentGroup->systemMatrix = new GraphScatteredMatrix();
    if (!currentGroup->systemRHVector) currentGroup->systemRHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
    if (!currentGroup->systemLHVector) currentGroup->systemLHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
    currentGroup->systemRHVector->reset();
    currentGroup->systemLHVector->reset();
    currentGroup->solutionVecId = core::MultiVecDerivId::null();
    currentGroup->needInvert = true;
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
//    serr << "setSystemMBKMatrix(" << mparams->mFactor() << ", " << mparams->bFactor() << ", " << mparams->kFactor() << ")" << sendl;
    createGroups(mparams);
    resetSystem();
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->systemMatrix->setMBKFacts(mparams);
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemRHVector(core::MultiVecDerivId v)
{
//    serr << "setSystemRHVector(" << v << ")" << sendl;
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->systemRHVector->set(v);
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemLHVector(core::MultiVecDerivId v)
{
//    serr << "setSystemLHVector(" << v << ")" << sendl;
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->solutionVecId = v;
        currentGroup->systemLHVector->set(v);
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::solveSystem()
{
//    serr << "solveSystem()" << sendl;
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        if (currentGroup->needInvert)
        {
//            serr << "->invert(M)" << sendl;
            this->invert(*currentGroup->systemMatrix);
//            serr << "<<invert(M)" << sendl;
            currentGroup->needInvert = false;
        }
//        serr << "->solve(M, " << currentGroup->systemLHVector->id()  << ", " << currentGroup->systemRHVector->id() << ")" << sendl;
        this->solve(*currentGroup->systemMatrix, *currentGroup->systemLHVector, *currentGroup->systemRHVector);
//        serr << "<<solve(M, " << currentGroup->systemLHVector->id()  << ", " << currentGroup->systemRHVector->id() << ")" << sendl;
    }
}

template<>
GraphScatteredMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::createMatrix()
{
    return new GraphScatteredMatrix();
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::deleteMatrix(GraphScatteredMatrix* v)
{
    delete v;
}

template<>
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::createPersistentVector()
{
    return new GraphScatteredVector(NULL,core::VecDerivId::null());
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::deletePersistentVector(GraphScatteredVector* v)
{
    delete v;
}

#ifdef SOFA_SMP
template<>
void MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::resetSystem()
{
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        if (!currentGroup->systemMatrix) currentGroup->systemMatrix = new GraphScatteredMatrix();
        if (!currentGroup->systemRHVector) currentGroup->systemRHVector = new ParallelGraphScatteredVector(NULL,core::VecDerivId::null());
        if (!currentGroup->systemLHVector) currentGroup->systemLHVector = new ParallelGraphScatteredVector(NULL,core::VecDerivId::null());
        currentGroup->systemRHVector->reset();
        currentGroup->systemLHVector->reset();
        currentGroup->solutionVecId = core::MultiVecDerivId::null();
        currentGroup->needInvert = true;
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::resizeSystem(int)
{
    if (!currentGroup->systemMatrix) currentGroup->systemMatrix = new GraphScatteredMatrix();
    if (!currentGroup->systemRHVector) currentGroup->systemRHVector = new ParallelGraphScatteredVector(NULL,core::VecDerivId::null());
    if (!currentGroup->systemLHVector) currentGroup->systemLHVector = new ParallelGraphScatteredVector(NULL,core::VecDerivId::null());
    currentGroup->systemRHVector->reset();
    currentGroup->systemLHVector->reset();
    currentGroup->solutionVecId = core::MultiVecDerivId::null();
    currentGroup->needInvert = true;
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    createGroups(mparams);
    resetSystem();
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->systemMatrix->setMBKFacts(mparams);
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::setSystemRHVector(core::MultiVecDerivId v)
{
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->systemRHVector->set(v);
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::setSystemLHVector(core::MultiVecDerivId v)
{
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->solutionVecId = v;
        currentGroup->systemLHVector->set(v);
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::solveSystem()
{
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        if (currentGroup->needInvert)
        {
            this->invert(*currentGroup->systemMatrix);
            currentGroup->needInvert = false;
        }
        this->solve(*currentGroup->systemMatrix, *currentGroup->systemLHVector, *currentGroup->systemRHVector);
    }
}

template<>
GraphScatteredMatrix* MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::createMatrix()
{
    return new GraphScatteredMatrix();
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::deleteMatrix(GraphScatteredMatrix* v)
{
    delete v;
}

template<>
ParallelGraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::createPersistentVector()
{
    return new ParallelGraphScatteredVector(NULL,core::VecDerivId::null());
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::deletePersistentVector(ParallelGraphScatteredVector* v)
{
    delete v;
}
#endif

/*
/////////// should be in an inl file ... ////////
template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
	const unsigned int Jrows = J.rowSize();
	const unsigned int Jcols = J.colSize();
	if (Jcols != this->systemMatrix->rowSize())
	{
		serr << "MatrixLinearSolver::addJMInvJt ERROR: incompatible J matrix size." << sendl;
		return false;
	}

	if (!Jrows) return false;
	//this->computeMinv();

	const typename JMatrix::LineConstIterator jitend = J.end();
	for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != jitend; ++jit1)
	{
	int row1 = jit1->first;
	for (typename JMatrix::LineConstIterator jit2 = jit1; jit2 != jitend; ++jit2)
	{
		int row2 = jit2->first;
		double acc = 0.0;
		for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(), i1end = jit1->second.end(); i1 != i1end; ++i1)
		{
			int col1 = i1->first;
			double val1 = i1->second;
			for (typename JMatrix::LElementConstIterator i2 = jit2->second.begin(), i2end = jit2->second.end(); i2 != i2end; ++i2)
			{
				int col2 = i2->first;
				double val2 = i2->second;
				acc += val1 * getMinvElement(col1,col2) * val2;
			}
		}
		acc *= fact;
		//sout << "W("<<row1<<","<<row2<<") += "<<acc<<" * "<<fact<<sendl;
		result.add(row1,row2,acc);
		if (row1!=row2)
			result.add(row2,row1,acc);
	}
	}
	return true;
}

template<class Matrix, class Vector>
bool MatrixLinearSolver<Matrix,Vector>::addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{

}
*/


template<>
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemBaseMatrix() { return NULL; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemRHBaseVector() { return NULL; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemLHBaseVector() { return NULL; }

#ifdef SOFA_SMP
template<>
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::getSystemBaseMatrix() { return NULL; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::getSystemRHBaseVector() { return NULL; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>::getSystemLHBaseVector() { return NULL; }
#endif

// Force template instantiation
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>;

#ifdef SOFA_SMP
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver<GraphScatteredMatrix,ParallelGraphScatteredVector>;
#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa
