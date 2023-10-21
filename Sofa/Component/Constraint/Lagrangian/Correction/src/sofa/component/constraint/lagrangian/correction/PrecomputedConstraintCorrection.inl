/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/constraint/lagrangian/correction/PrecomputedConstraintCorrection.h>

#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>

#include <sofa/component/odesolver/backward/EulerImplicitSolver.h>

#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>

#include <sofa/core/behavior/RotationFinder.h>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/type/Quat.h>

#include <sofa/simulation/fwd.h>

#include <fstream>
#include <sstream>
#include <list>
#include <iomanip>
#include <sofa/helper/system/FileSystem.h>

//#define NEW_METHOD_UNBUILT

namespace sofa::component::constraint::lagrangian::correction
{

template<class DataTypes>
PrecomputedConstraintCorrection<DataTypes>::PrecomputedConstraintCorrection(sofa::core::behavior::MechanicalState<DataTypes> *mm)
    : Inherit(mm)
    , m_rotations(initData(&m_rotations, false, "rotations", ""))
    , m_restRotations(initData(&m_restRotations, false, "restDeformations", ""))
    , recompute(initData(&recompute, false, "recompute", "if true, always recompute the compliance"))
    , debugViewFrameScale(initData(&debugViewFrameScale, 1.0_sreal, "debugViewFrameScale", "Scale on computed node's frame"))
    , f_fileCompliance(initData(&f_fileCompliance, "fileCompliance", "Precomputed compliance matrix data file"))
    , fileDir(initData(&fileDir, "fileDir", "If not empty, the compliance will be saved in this repertory"))
    , invM(nullptr)
    , appCompliance(nullptr)
    , nbRows(0), nbCols(0), dof_on_node(0), nbNodes(0)
{
    this->addAlias(&f_fileCompliance, "filePrefix");
}

template<class DataTypes>
PrecomputedConstraintCorrection<DataTypes>::~PrecomputedConstraintCorrection()
{
    releaseInverse(invName, invM);
}



//////////////////////////////////////////////////////////////////////////
//   Precomputation of the Constraint Correction for all type of data
//////////////////////////////////////////////////////////////////////////

template<class DataTypes>
typename PrecomputedConstraintCorrection<DataTypes>::InverseStorage* PrecomputedConstraintCorrection<DataTypes>::getInverse(std::string name)
{
    std::map< std::string, InverseStorage >& registry = getInverseMap();
    InverseStorage* m = &(registry[name]);
    ++m->nbref;
    return m;
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::releaseInverse(std::string name, InverseStorage* inv)
{
    if (inv == nullptr) return;
    std::map< std::string, InverseStorage >& registry = getInverseMap();
    if (--inv->nbref == 0)
    {
        if (inv->data) delete[] inv->data;
        registry.erase(name);
    }
}


struct ConstraintActivation { bool acc, vel, pos; };


template<class DataTypes>
std::string PrecomputedConstraintCorrection<DataTypes>::buildFileName()
{
    const SReal dt = this->getContext()->getDt();
    const std::string name = this->getContext()->getName();

    std::stringstream ss;
    ss << name << "-" << nbRows << "-" << dt <<".comp";

    return ss.str();
}



template<class DataTypes>
bool PrecomputedConstraintCorrection<DataTypes>::loadCompliance(std::string fileName)
{
    // Try to load from memory
    msg_info() << "Try to load compliance from memory " << fileName ;

    invM = getInverse(fileName);
    dimensionAppCompliance = nbRows;

    if (invM->data == nullptr)
    {
        // Try to load from file
        msg_info() << "Try to load compliance from : " << fileName ;

        std::string dir = fileDir.getValue();
        if (!dir.empty())
        {
            const std::string path = helper::system::FileSystem::append(dir, fileName);
            std::ifstream compFileIn(path, std::ifstream::binary);
            if (compFileIn.is_open())
            {
                invM->data = new Real[nbRows * nbCols];

                msg_info() << "File " << path << " found. Loading..." ;

                compFileIn.read((char*)invM->data, nbCols * nbRows * sizeof(Real));
                compFileIn.close();

                return true;
            }
            else
                return false;
        }
        else if (recompute.getValue() == false)
        {
            std::stringstream ss;
            if (sofa::helper::system::DataRepository.findFile(fileName, "", &ss))
            {
                invM->data = new Real[nbRows * nbCols];

                std::ifstream compFileIn(fileName.c_str(), std::ifstream::binary);

                msg_info() << "File " << fileName << " found. Loading..." ;

                compFileIn.read((char*)invM->data, nbCols * nbRows * sizeof(Real));
                compFileIn.close();

                return true;
            }
            else
            {
                msg_info() << sofa::helper::removeTrailingCharacters(ss.str(), {'\n', '\r'})
                    << ". Compliance will be pre-computed and saved into a file";
            }
        }

        return false;
    }

    return true;
}



template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::saveCompliance(const std::string& fileName)
{    
    std::string filePathInSofaShare;
    const std::string dir = fileDir.getValue();
    if (!dir.empty())
    {
        filePathInSofaShare = helper::system::FileSystem::append(dir, fileName);
    }
    else
    {
        filePathInSofaShare = helper::system::FileSystem::append(
            sofa::helper::system::DataRepository.getFirstPath(), fileName);
    }

    const bool printLog = this->f_printLog.getValue();
    this->f_printLog.setValue(true);
    msg_info() << "Compliance file has been saved in " << filePathInSofaShare << ". Load this file using fileCompliance if you don't want to recompute the compliance matrice at next start.";
    this->f_printLog.setValue(printLog);

    std::ofstream compFileOut(filePathInSofaShare.c_str(), std::fstream::out | std::fstream::binary);
    compFileOut.write((char*)invM->data, nbCols * nbRows * sizeof(Real));
    compFileOut.close();
}



template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::bwdInit()
{
    Inherit::init();

    const VecDeriv& v0 = this->mstate->read(core::ConstVecDerivId::velocity())->getValue();

    nbNodes = v0.size();

    if (nbNodes == 0)
    {
        msg_error() << "No degree of freedom" ;
        return;
    }

    dof_on_node = v0[0].size();

    nbRows = nbNodes * dof_on_node;
    nbCols = nbNodes * dof_on_node;

    const SReal dt = this->getContext()->getDt();

    invName = f_fileCompliance.getFullPath();
    bool complianceLoaded = false;
    if (!invName.empty())
    {
        complianceLoaded = loadCompliance(invName);
        if (!complianceLoaded) 
        {
            msg_error() << "A fileCompliance was given at path: " << invName << ", but could not be loaded.";
            invName = buildFileName();
        }
    }
    else
    {
        invName = buildFileName();
        complianceLoaded = loadCompliance(invName);
    }

    if (!complianceLoaded)
    {
        msg_info() << "Compliance being built";

        // Buffer Allocation
        invM->data = new Real[nbRows * nbCols];

        // for the intial computation, the gravity has to be put at 0
        const sofa::type::Vec3& gravity = this->getContext()->getGravity();

        static constexpr sofa::type::Vec3 gravity_zero(0_sreal, 0_sreal, 0_sreal);
        this->getContext()->setGravity(gravity_zero);

        sofa::component::odesolver::backward::EulerImplicitSolver* eulerSolver;
        sofa::component::linearsolver::iterative::CGLinearSolver< sofa::component::linearsolver::GraphScatteredMatrix, sofa::component::linearsolver::GraphScatteredVector >* cgLinearSolver;
        core::behavior::LinearSolver* linearSolver;

        this->getContext()->get(eulerSolver);
        this->getContext()->get(cgLinearSolver);
        this->getContext()->get(linearSolver);

        if (eulerSolver && cgLinearSolver)
        {
            msg_info() << "use EulerImplicitSolver & CGLinearSolver" ;
        }
        else if (eulerSolver && linearSolver)
        {
            msg_info() << "use EulerImplicitSolver & LinearSolver";
        }
        else if(eulerSolver)
        {
            msg_info() << "use EulerImplicitSolver";
        }
        else
        {
            msg_error() << "PrecomputedContactCorrection must be associated with EulerImplicitSolver+LinearSolver for the precomputation\nNo Precomputation" ;
            return;
        }

        // Change the solver parameters
        Real buf_tolerance = 0, buf_threshold = 0;
        unsigned int	   buf_maxIter = 0;

        if (cgLinearSolver)
        {
            buf_tolerance = cgLinearSolver->d_tolerance.getValue();
            buf_maxIter   = cgLinearSolver->d_maxIter.getValue();
            buf_threshold = cgLinearSolver->d_smallDenominatorThreshold.getValue();

            cgLinearSolver->d_tolerance.setValue(Real(1e-20));
            cgLinearSolver->d_maxIter.setValue(5000u);
            cgLinearSolver->d_smallDenominatorThreshold.setValue(Real(1e-35));
        }


        helper::ReadAccessor< Data< VecCoord > > rposData = *this->mstate->read(core::ConstVecCoordId::position());
        const VecCoord prev_pos = rposData.ref();

        helper::WriteAccessor< Data< VecDeriv > > velocityData = *this->mstate->write(core::VecDerivId::velocity());
        VecDeriv& velocity = velocityData.wref();
        const VecDeriv prev_velocity = velocity;

        helper::WriteAccessor< Data< VecDeriv > > forceData = *this->mstate->write(core::VecDerivId::externalForce());
        VecDeriv& force = forceData.wref();
        force.clear();
        force.resize(nbNodes);

        /// christian : it seems necessary to called the integration one time for initialization
        /// (avoid to have a line of 0 at the top of the matrix)
        if (eulerSolver)
        {
            eulerSolver->solve(core::execparams::defaultInstance(), dt, core::VecCoordId::position(), core::VecDerivId::velocity());
        }

        Deriv unitary_force;

        std::stringstream tmpStr;
        for (unsigned int f = 0; f < nbNodes; f++)
        {
            tmpStr.precision(2);
            tmpStr << "Precomputing constraint correction : " << std::fixed << (float)f / (float)nbNodes * 100.0f << " %   " << '\xd';

            for (unsigned int i = 0; i < dof_on_node; i++)
            {
                unitary_force.clear();
                unitary_force[i] = 1.0;

                force[f] = unitary_force;

                // Reset positions and velocities
                velocity.clear();
                velocity.resize(nbNodes);

                // Actualize ref to the position vector ; it seems it is changed at every eulerSolver->solve()
                helper::WriteOnlyAccessor< Data< VecCoord > > wposData = *this->mstate->write(core::VecCoordId::position());
                VecCoord& pos = wposData.wref();

                for (unsigned int n = 0; n < nbNodes; n++)
                    pos[n] = prev_pos[n];

                SReal fact = 1.0_sreal / dt; // christian : it is not a compliance... but an admittance that is computed !

                if (eulerSolver)
                {
                    fact *= eulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

                    eulerSolver->solve(core::execparams::defaultInstance(), dt, core::VecCoordId::position(), core::VecDerivId::velocity());

                    if (linearSolver)
                        linearSolver->freezeSystemMatrix(); // do not recompute the matrix for the rest of the precomputation
                }

                for (unsigned int v = 0; v < nbNodes; v++)
                {
                    for (unsigned int j = 0; j < dof_on_node; j++)
                    {
                        invM->data[(v * dof_on_node + j) * nbCols + (f * dof_on_node + i) ] = (Real)(fact * velocity[v][j]);
                    }
                }
            }

            unitary_force.clear();
            force[f] = unitary_force;
            msg_info() << tmpStr.str();
        }        

        // Do not recompute the matrix for the rest of the precomputation
        if (linearSolver)
            linearSolver->freezeSystemMatrix();

        saveCompliance(invName);

        // Restore gravity
        this->getContext()->setGravity(gravity);

        // Restore linear solver parameters
        if (cgLinearSolver)
        {
            cgLinearSolver->d_tolerance.setValue(buf_tolerance);
            cgLinearSolver->d_maxIter.setValue(buf_maxIter);
            cgLinearSolver->d_smallDenominatorThreshold.setValue(buf_threshold);
        }

        // Retore velocity
        for (unsigned int i = 0; i < velocity.size(); i++)
            velocity[i] = prev_velocity[i];

        helper::WriteOnlyAccessor< Data< VecCoord > > wposData = *this->mstate->write(core::VecCoordId::position());
        VecCoord& pos = wposData.wref();

        // Restore position
        for (unsigned int i = 0; i < pos.size(); i++)
            pos[i] = prev_pos[i];
    }

    appCompliance = invM->data;

    // Optimisation for the computation of W
    _indexNodeSparseCompliance.resize(v0.size());
}


template< class DataTypes >
void PrecomputedConstraintCorrection< DataTypes >::addComplianceInConstraintSpace(const sofa::core::ConstraintParams *cparams, sofa::linearalgebra::BaseMatrix* W)
{
    m_activeDofs.clear();

	const MatrixDeriv& c = cparams->readJ(this->mstate)->getValue();

    SReal factor = 1.0_sreal;

    switch (cparams->constOrder())
    {
    case core::ConstraintOrder::POS_AND_VEL :
    case core::ConstraintOrder::POS :
        break;

    case core::ConstraintOrder::ACC :
    case core::ConstraintOrder::VEL :
        factor = 1.0 / this->getContext()->getDt(); // @TODO : Consistency between ODESolver & Compliance and/or Admittance computation
        break;

    default :
        break;
    }


    /////////// The constraints are modified using a rotation value at each node/////
    if (m_rotations.getValue())
        rotateConstraints(false);


    /////////// Which node are involved with the contact ? /////

    const unsigned int noSparseComplianceSize = _indexNodeSparseCompliance.size();

    for (unsigned int i = 0; i < noSparseComplianceSize; ++i)
    {
        _indexNodeSparseCompliance[i] = -1;
    }

    int nActiveDof = 0;
    unsigned int nbConstraints = 0;

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        MatrixDerivColConstIterator colItEnd = rowIt.end();

        for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int dof = colIt.index();
            m_activeDofs.push_back(dof);

            if (_indexNodeSparseCompliance[dof] != 0)
            {
                ++nActiveDof;
                _indexNodeSparseCompliance[dof] = 0;
            }
        }

        nbConstraints++;
    }

    m_activeDofs.sort();
    m_activeDofs.unique();

    unsigned int offset, offset2;
    unsigned int ii,jj, it;
    Deriv Vbuf;
    it = 0;

    _sparseCompliance.resize(nActiveDof * nbConstraints);

    for (int NodeIdx = 0; NodeIdx < (int)noSparseComplianceSize; ++NodeIdx)
    {
        if (_indexNodeSparseCompliance[NodeIdx] == -1)
            continue;

        _indexNodeSparseCompliance[NodeIdx] = it;

        for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
        {
            Vbuf.clear();

            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const Deriv n2 = colIt.val();
                offset = dof_on_node * (NodeIdx * nbCols +  colIt.index());

                for (ii = 0; ii < dof_on_node; ii++)
                {
                    offset2 = offset + ii * nbCols;

                    for (jj = 0; jj < dof_on_node; jj++)
                    {
                        Vbuf[ii] += appCompliance[offset2 + jj] * n2[jj];
                    }
                }
            }

            _sparseCompliance[it] = Vbuf;
            it++;
        }
    }

    unsigned int curConstraint = 0;

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        int indexCurRowConst = rowIt.index();

        MatrixDerivColConstIterator colItEnd = rowIt.end();

        for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const Deriv n1 = colIt.val();

            const unsigned int temp = (unsigned int) _indexNodeSparseCompliance[colIt.index()];

            unsigned int curColConst = curConstraint;

            for (MatrixDerivRowConstIterator rowIt2 = rowIt; rowIt2 != rowItEnd; ++rowIt2)
            {
                int indexCurColConst = rowIt2.index();
                Real w = _sparseCompliance[temp + curColConst] * n1 * factor;

                W->add(indexCurRowConst, indexCurColConst, w);

                if (indexCurRowConst != indexCurColConst)
                    W->add(indexCurColConst, indexCurRowConst, w);

                curColConst++;
            }
        }
        curConstraint++;
    }
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::computeMotionCorrection(const core::ConstraintParams* /*cparams*/, core::MultiVecDerivId dx, core::MultiVecDerivId f)
{
    computeDx(*dx[this->getMState()].write(), *f[this->getMState()].read(), m_activeDofs);
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::computeDx(Data<VecDeriv>& dx_d , const Data< VecDeriv > &f_d, const std::list< int > &activeDofs)
{
    const VecDeriv& force = f_d.getValue();
    VecDeriv& dx = *dx_d.beginEdit();

    dx.clear();
    dx.resize(force.size());

    std::list<int>::const_iterator IterateurListe;
    unsigned int i, offset, offset2;

    for (IterateurListe = activeDofs.begin(); IterateurListe != activeDofs.end(); ++IterateurListe)
    {
        int f = (*IterateurListe);

        for (i = 0; i < dof_on_node; i++)
        {
            Fbuf[i] = force[f][i];
        }

        for (unsigned int v = 0 ; v < dx.size() ; v++)
        {
            offset =  v * dof_on_node * nbCols + f * dof_on_node;
            for (unsigned int j = 0; j < dof_on_node; j++)
            {
                offset2 = offset + j * nbCols;
                DXbuf = 0.0;

                for (i = 0; i < dof_on_node; i++)
                {
                    DXbuf += appCompliance[ offset2 + i ] * Fbuf[i];
                }

                dx[v][j] += DXbuf;
            }
        }
    }

    dx_d.endEdit();
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::applyMotionCorrection(const sofa::core::ConstraintParams *cparams
        , sofa::Data< VecCoord > &x_d, sofa::Data< VecDeriv > &v_d, sofa::Data< VecDeriv > & dx_d, const sofa::Data< VecDeriv > & correction_d )
{
    VecCoord& x = *x_d.beginEdit();
    VecDeriv& v = *v_d.beginEdit();

    const VecDeriv& correction = correction_d.getValue();

    auto dx = sofa::helper::getWriteAccessor(dx_d);

    const VecCoord& x_free = cparams->readX(this->mstate)->getValue();
    const VecDeriv& v_free = cparams->readV(this->mstate)->getValue();

    const SReal invDt = 1.0_sreal / this->getContext()->getDt();

    if (m_rotations.getValue())
        rotateResponse();

    for (unsigned int i=0; i< dx.size(); i++)
    {
        x[i] = x_free[i];
        v[i] = v_free[i];

        x[i] += correction[i];
        v[i] += correction[i] * invDt;
        dx[i] = correction[i];
    }

    x_d.endEdit();
    v_d.endEdit();
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::applyPositionCorrection(const sofa::core::ConstraintParams *cparams, 
    sofa::Data< VecCoord > &x_d, sofa::Data< VecDeriv > &dx_d, const sofa::Data< VecDeriv > & correction_d)
{
    VecCoord& x = *x_d.beginEdit();

    const VecDeriv& correction = correction_d.getValue();

    auto dx = sofa::helper::getWriteAccessor(dx_d);

    const VecCoord& x_free = cparams->readX(this->mstate)->getValue();

    if (m_rotations.getValue())
        rotateResponse();

    for (unsigned int i=0; i< dx.size(); i++)
    {
        x[i]  = x_free[i] + correction[i];
        dx[i] = correction[i];
    }

    x_d.endEdit();
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::applyVelocityCorrection(const sofa::core::ConstraintParams *cparams, 
    sofa::Data< VecDeriv > &v_d, sofa::Data< VecDeriv > &dv_d, const sofa::Data<VecDeriv>& correction_d)
{
    const VecDeriv& correction = correction_d.getValue();
    auto dv = sofa::helper::getWriteAccessor(dv_d);
    VecDeriv& v = *v_d.beginEdit();

    const VecDeriv& dx = this->mstate->read(core::VecDerivId::dx())->getValue();
    const VecDeriv& v_free = cparams->readV(this->mstate)->getValue();

    const SReal invDt = 1.0_sreal / this->getContext()->getDt();

    if (m_rotations.getValue())
        rotateResponse();

    for (unsigned int i=0; i< dx.size(); i++)
    {
        dv[i] = correction[i] * invDt;
        v[i] = v_free[i] + dv[i];
    }

    v_d.endEdit();
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::applyContactForce(const linearalgebra::BaseVector *f)
{
    helper::WriteAccessor<Data<VecDeriv> > forceData = *this->mstate->write(core::VecDerivId::force());
    helper::WriteAccessor<Data<VecDeriv> > dxData    = *this->mstate->write(core::VecDerivId::dx());
    helper::WriteAccessor<Data<VecCoord> > xData     = *this->mstate->write(core::VecCoordId::position());
    helper::WriteAccessor<Data<VecDeriv> > vData     = *this->mstate->write(core::VecDerivId::velocity());
    VecDeriv& force = forceData.wref();
    VecDeriv& dx = dxData.wref();
    VecCoord& x = xData.wref();
    VecDeriv& v = vData.wref();

    const VecDeriv& v_free = this->mstate->read(core::ConstVecDerivId::freeVelocity())->getValue();
    const VecCoord& x_free = this->mstate->read(core::ConstVecCoordId::freePosition())->getValue();
    const MatrixDeriv& c = this->mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

    const SReal dt = this->getContext()->getDt();

    dx.clear();
    dx.resize(v.size());

    force.clear();
    force.resize(x_free.size());

    std::list<int> activeDofs;

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        Real fC1 = (Real)f->element(rowIt.index());

        if (fC1 != 0.0)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                unsigned int dof = colIt.index();
                force[dof] += colIt.val() * fC1;
                activeDofs.push_back(dof);
            }
        }
    }

    activeDofs.sort();
    activeDofs.unique();

    unsigned int offset, offset2;
    for (const auto dofId : activeDofs)
    {
        for (unsigned int i=0; i< dof_on_node; i++)
        {
            Fbuf[i] = force[dofId][i];
        }

        for(unsigned int i = 0 ; i < dx.size() ; i++)
        {
            offset =  i * dof_on_node * nbCols + dofId * dof_on_node;
            for (unsigned int j=0; j< dof_on_node; j++)
            {
                offset2 = offset+ j*nbCols;
                DXbuf=0.0;
                for (unsigned int k = 0; k < dof_on_node; k++)
                {
                    DXbuf += appCompliance[ offset2 + k ] * Fbuf[k];
                }
                dx[i][j]+=DXbuf;
            }
        }
    }

    force.clear();
    force.resize(x_free.size());

    if (m_rotations.getValue())
        rotateResponse();

    for (unsigned int i=0; i< dx.size(); i++)
    {
        x[i] = x_free[i];
        v[i] = v_free[i];

        x[i] += dx[i];
        v[i] += dx[i] * (1/dt);
    }
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::getComplianceMatrix(linearalgebra::BaseMatrix* m) const
{
    m->resize(dimensionAppCompliance,dimensionAppCompliance);

    for (unsigned int l = 0; l < dimensionAppCompliance; ++l)
    {
        for (unsigned int c = 0; c < dimensionAppCompliance; ++c)
        {
            m->set(l, c, appCompliance[l * dimensionAppCompliance + c]);
        }
    }
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::resetContactForce()
{
    helper::WriteAccessor<Data<VecDeriv> > forceData = *this->mstate->write(core::VecDerivId::force());
    VecDeriv& force = forceData.wref();
    for( unsigned i=0; i<force.size(); ++i )
        force[i] = Deriv();
}


template< class DataTypes >
void PrecomputedConstraintCorrection< DataTypes >::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels() || !m_rotations.getValue())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    using sofa::core::behavior::RotationFinder;

    // we draw the rotations associated to each node //

    const simulation::Node *node = dynamic_cast< simulation::Node* >(this->getContext());

    RotationFinder< DataTypes > * rotationFinder = nullptr;

    if (node != nullptr)
    {
        rotationFinder = node->get< RotationFinder< DataTypes > > ();
        if (rotationFinder == nullptr)
        {
            return;
        }
    }

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const auto& rotations = rotationFinder->getRotations();
    for (unsigned int i=0; i< x.size(); i++)
    {
        sofa::type::Matrix3 RotMat;

        for (unsigned int a=0; a<3; a++)
        {
            for (unsigned int b=0; b<3; b++)
            {
                RotMat[a][b] = rotations[i](a,b);
            }
        }

        sofa::type::Quat<SReal> q;
        q.fromMatrix(RotMat);
        vparams->drawTool()->drawFrame(DataTypes::getCPos(x[i]), q, sofa::type::Vec3f(this->debugViewFrameScale.getValue(),this->debugViewFrameScale.getValue(),this->debugViewFrameScale.getValue()));

    }


}


template< class DataTypes >
void PrecomputedConstraintCorrection< DataTypes >::rotateConstraints(bool back)
{
    using sofa::core::behavior::RotationFinder;

    helper::WriteAccessor<Data<MatrixDeriv> > cData = *this->mstate->write(core::MatrixDerivId::constraintJacobian());
    MatrixDeriv& c = cData.wref();

    const simulation::Node *node = dynamic_cast< simulation::Node * >(this->getContext());

    RotationFinder< DataTypes >* rotationFinder = nullptr;

    if (node != nullptr)
    {
        rotationFinder = node->get< RotationFinder< DataTypes > > ();
        if (rotationFinder == nullptr)
        {
            msg_warning() << "No rotation defined : only applicable for components implementing RotationFinder!";
            return;
        }
    }
    else
    {
        msg_error() << "Error getting context in method: PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateConstraints(false)";
        return;
    }

    // on fait tourner les normales (en les ramenant dans le "pseudo" repere initial) //
    auto rowItEnd = c.end();
    const auto& rotations = rotationFinder->getRotations();

    for (auto rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        [[maybe_unused]] auto rowWrite = c.writeLine(rowIt.index());
        auto colItEnd = rowIt.end();

        for (auto colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            Deriv n = colIt.val();
            const int localRowNodeIdx = colIt.index();

            // rotationFinder has been defined
            auto Ri = rotations[localRowNodeIdx];

            if(!back)
                Ri.transpose();

            // on passe les normales du repere global au repere local
            const Deriv n_i = Ri * n;
            n.x() =  n_i.x();
            n.y() =  n_i.y();
            n.z() =  n_i.z();
        }
    }
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::rotateResponse()
{
    const simulation::Node *node = dynamic_cast<simulation::Node *>(this->getContext());

    using sofa::core::behavior::RotationFinder;

    RotationFinder< DataTypes >* rotationFinder = nullptr;

    if (node != nullptr)
    {
        rotationFinder = node->get< RotationFinder< DataTypes > > ();
        if (rotationFinder == nullptr)
        {
            msg_warning() << "No rotation defined : only applicable for components implementing RotationFinder!";
            return;
        }
    }
    else
    {
        msg_error() << "Error getting context in method: PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateConstraints(false)";
        return;
    }

    helper::WriteAccessor<Data<VecDeriv> > dxData = *this->mstate->write(core::VecDerivId::dx());
    VecDeriv& dx = dxData.wref();
    const auto& rotations = rotationFinder->getRotations();

    for(unsigned int j = 0; j < dx.size(); j++)
    {
        // on passe les deplacements du repere local au repere global
        dx[j] = rotations[j] * dx[j];
    }
}


// new API for non building the constraint system during solving process //
template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::resetForUnbuiltResolution(SReal* f, std::list<unsigned int>& /*renumbering*/)
{
    constraint_force = f;
    const MatrixDeriv& c = this->mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

#ifdef NEW_METHOD_UNBUILT
    constraint_D.clear();
    constraint_D.resize(this->mstate->getSize());

    constraint_F.clear();
    constraint_F.resize(this->mstate->getSize());

    constraint_dofs.clear();

    bool error_message_not_displayed=true;
#endif

    /////////// The constraints are modified using a rotation value at each node/////
    if (m_rotations.getValue())
        rotateConstraints(false);

    unsigned int nbConstraints = 0;

    /////////// Which node are involved with the contact ?/////
    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        MatrixDerivColConstIterator colItEnd = rowIt.end();

        for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            constraint_dofs.push_back(colIt.index());
        }

        nbConstraints++;
    }

    constraint_dofs.sort();
    constraint_dofs.unique();

    id_to_localIndex.clear();
    localIndex_to_id.clear();
    active_local_force.clear();
    unsigned int cpt = 0;

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        unsigned int cId = rowIt.index();

        if (cId >= id_to_localIndex.size())
            id_to_localIndex.resize(cId + 1, -1);

        msg_error_when(id_to_localIndex[cId] != -1) << "duplicate entry in constraints for id " << cId << " : " << id_to_localIndex[cId] << " + " << cpt;

        id_to_localIndex[cId] = cpt;
        localIndex_to_id.push_back(cId);

        if(fabs(f[cId]) > std::numeric_limits<SReal>::epsilon())
            active_local_force.push_back(cpt);

#ifdef NEW_METHOD_UNBUILT  // Fill constraint_F => provide the present constraint forces
        SReal fC = f[rowIt.index()];

        if (fC != 0.0)
        {
            if(error_message_not_displayed)
            {
                msg_error() << "Initial_guess not supported yet in unbuilt mode with NEW_METHOD_UNBUILT!=> PUT F to 0";
                error_message_not_displayed = false;
            }

            f[rowIt.index()] = 0.0;

            /*
            ConstraintIterator itConstraint;
            std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[i].data();

            for (itConstraint=iter.first;itConstraint!=iter.second;itConstraint++)
            {
                unsigned int dof = itConstraint->first;
                Deriv n = itConstraint->second;
                constraint_F[dof] +=n * fC;

                // TODO : remplacer pour faire + rapide !!
            //	setConstraintDForce(&fC, (int)c, (int)c, true);

            }
            */
        }
#endif
        cpt++;
    }

#ifndef NEW_METHOD_UNBUILT

    unsigned int offset, offset2;
    Deriv Vbuf;
    unsigned int it = 0;

    _sparseCompliance.resize(constraint_dofs.size() * nbConstraints);

    const auto dofsItEnd = constraint_dofs.end();

    for (auto dofsIt = constraint_dofs.begin(); dofsIt != dofsItEnd; ++dofsIt)
    {
        const int NodeIdx = (*dofsIt);
        _indexNodeSparseCompliance[NodeIdx] = it;

        for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
        {
            Vbuf.clear();

            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                offset = dof_on_node * (NodeIdx * nbCols +  colIt.index());

                for (unsigned int ii = 0; ii < dof_on_node; ii++)
                {
                    offset2 = offset + ii *nbCols;

                    for (unsigned int jj = 0; jj < dof_on_node; jj++)
                    {
                        Vbuf[ii] += appCompliance[offset2 + jj] * colIt.val()[jj];
                    }
                }
            }

            _sparseCompliance[it] = Vbuf;
            it++;
        }
    }

    localW.resize(nbConstraints, nbConstraints);

    unsigned int curRowConst = 0;

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        MatrixDerivColConstIterator colItEnd = rowIt.end();

        for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const Deriv n1 = colIt.val();

            const unsigned int temp = (unsigned int) _indexNodeSparseCompliance[colIt.index()];

            unsigned int curColConst = curRowConst;

            for (MatrixDerivRowConstIterator rowIt2 = rowIt; rowIt2 != rowItEnd; ++rowIt2)
            {
                Real w = _sparseCompliance[temp + curColConst] * n1;

                localW.add(curRowConst, curColConst, w);

                if (curRowConst != curColConst)
                    localW.add(curColConst, curRowConst, w);

                curColConst++;
            }
        }
        curRowConst++;
    }
#endif
}


template<class DataTypes>
bool PrecomputedConstraintCorrection<DataTypes>::hasConstraintNumber(int index)
{
    return ((Size)index) < id_to_localIndex.size() && id_to_localIndex[index] >= 0;
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::addConstraintDisplacement(SReal* d, int begin, int end)
{
#ifdef NEW_METHOD_UNBUILT

    const MatrixDeriv& c = *this->mstate->getC();

    for (int i = begin; i <= end; i++)
    {
        int cId = id_to_localIndex[i];

        MatrixDerivRowConstIterator rowIt = c.readLine(cId);

        if (rowIt != c.end())
        {
            SReal dc = d[i];

            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                dc += colIt.val() * constraint_D[colIt.index()];
            }

            d[i] = dc;
        }
    }

#else
    const std::list<unsigned int>::iterator itBegin = active_local_force.begin(), itEnd = active_local_force.end();

    for (int i = begin; i <= end; i++)
    {
        int c = id_to_localIndex[i];
        double dc = d[i];

        for (auto it = itBegin; it != itEnd; ++it)
        {
            const int id = localIndex_to_id[*it];
            dc += localW.element(c, *it) * constraint_force[id];
        }

        d[i] = dc;
    }
#endif
}

template<class DataTypes>
#ifdef NEW_METHOD_UNBUILT
void PrecomputedConstraintCorrection<DataTypes>::setConstraintDForce(SReal * df, int begin, int end, bool update)
#else
void PrecomputedConstraintCorrection<DataTypes>::setConstraintDForce(SReal * /*df*/, int begin, int end, bool update)
#endif
{
#ifdef NEW_METHOD_UNBUILT

    /// set a force difference on a set of constraints (between constraint number "begin" and constraint number "end"
    /// if update is false, do nothing
    /// if update is true, it computes the displacements due to this delta of force.
    /// As the contact are uncoupled, a displacement is obtained only on dof involved with the constraints

    const MatrixDeriv& c = *this->mstate->getC();

    if (!update)
        return;

    unsigned int offset, offset2;

    for (int i = begin; i <= end; i++)
    {
        int cId = id_to_localIndex[i];

        MatrixDerivRowConstIterator rowIt = c.readLine(cId);

        if (rowIt != c.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                Deriv n = colIt.val();
                unsigned int dof = colIt.index();

                constraint_F[dof] += n * df[i];

                for (unsigned int j = 0; j < dof_on_node; j++)
                {
                    Fbuf[j] = n[j] * df[i];
                }

                std::list< int >::const_iterator dofsItEnd = constraint_dofs.end();

                for (std::list< int >::const_iterator dofsIt = constraint_dofs.begin(); dofsIt != dofsItEnd; ++dofsIt)
                {
                    int dof2 = *dofsIt;
                    offset = dof2 * dof_on_node * nbCols + dof * dof_on_node;

                    for (unsigned int j = 0; j < dof_on_node; j++)
                    {
                        offset2 = offset + j * nbCols;
                        DXbuf = 0.0;
                        for (unsigned int k = 0; k < dof_on_node; k++)
                        {
                            DXbuf += appCompliance[ offset2 + k ] * Fbuf[k];
                        }

                        constraint_D[dof2][j] += DXbuf;
                    }
                }
            }
        }
    }
#else
    if(!update)
        return;

    /// fill a local table of active forces (non-null forces)
    for (int i = begin; i <= end; i++)
    {
        int c = id_to_localIndex[i];
        active_local_force.push_back(c);
    }
    active_local_force.sort();
    active_local_force.unique();
#endif
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::getBlockDiagonalCompliance(linearalgebra::BaseMatrix* W, int begin, int end)
{
#ifdef NEW_METHOD_UNBUILT

    const MatrixDeriv& c = *this->mstate->getC();
    int numLocalConstraints = 0;

    std::list<int> localActiveDof;
    std::vector<int> constraintLocalID;

    for (int i = begin; i <= end; i++)
    {
        numLocalConstraints++;
        int cId = id_to_localIndex[i];
        constraintLocalID.push_back(i);

        MatrixDerivRowConstIterator rowIt = c.readLine(cId);

        if (rowIt != c.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                localActiveDof.push_back(colIt.index());
            }
        }
    }

    localActiveDof.sort();
    localActiveDof.unique();

    unsigned int offset, offset2;
    Deriv Vbuf;
    int it = 0;
    int it_localActiveDof = 0;

    _sparseCompliance.resize(localActiveDof.size() * numLocalConstraints);

    std::list< int >::const_iterator dofsItEnd = localActiveDof.end();

    for (std::list< int >::const_iterator dofsIt = localActiveDof.begin(); dofsIt != dofsItEnd; ++dofsIt)
    {
        int dof1 = (*dofsIt);
        _indexNodeSparseCompliance[dof1] = it_localActiveDof;
        it_localActiveDof++;

        for (int i = begin; i <= end; i++)
        {
            int cId = id_to_localIndex[i];

            Vbuf.clear();  // displacement obtained on the active node  dof 1  when apply contact force 1 on constraint c

            MatrixDerivRowConstIterator rowIt = c.readLine(cId);

            if (rowIt != c.end())
            {
                MatrixDerivColConstIterator colItEnd = rowIt.end();

                for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
                {
                    const Deriv n2 = colIt.val();

                    offset = dof_on_node * (dof1 * nbCols +  colIt.index());

                    for (unsigned int ii = 0; ii < dof_on_node; ii++)
                    {
                        offset2 = offset + ii * nbCols;

                        for (unsigned int jj = 0; jj < dof_on_node; jj++)
                        {
                            Vbuf[ii] += appCompliance[offset2 + jj] * n2[jj];
                        }
                    }
                }
            }

            _sparseCompliance[it] = Vbuf;   // [it = numLocalConstraints *
            it++;
        }
    }
    it = 0;

    for (int i = begin; i <= end; i++)
    {
        int c1 = id_to_localIndex[i];

        MatrixDerivRowConstIterator rowIt = c.readLine(c1);

        if (rowIt != c.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const Deriv n1 = colIt.val();
                unsigned int c1_loc = (unsigned int) _indexNodeSparseCompliance[colIt.index()];

                for(int c2_loc = it; c2_loc < numLocalConstraints; c2_loc++)
                {
                    int id2 = constraintLocalID[c2_loc];

                    auto w = n1 * _sparseCompliance[c1_loc * numLocalConstraints + c2_loc];

                    W->add(i, id2, w);

                    if (i != id2)
                    {
                        W->add(id2, i, w);
                    }
                }
            }

            it++;
        }
    }

#else

    for (int id1 = begin; id1<=end; id1++)
    {
        int c1 = id_to_localIndex[id1];
        for (int id2 = id1; id2<=end; id2++)
        {
            int c2 = id_to_localIndex[id2];
            Real w = (Real)localW.element(c1,c2);

            W->add(id1, id2, w);
            if (id1 != id2)
                W->add(id2, id1, w);
        }
    }

#endif

}

} //namespace sofa::component::constraint::lagrangian::correction
