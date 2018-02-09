/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_UNCOUPLEDCONSTRAINTCORRECTION_INL
#define SOFA_COMPONENT_CONSTRAINTSET_UNCOUPLEDCONSTRAINTCORRECTION_INL

#include "UncoupledConstraintCorrection.h"

#include <sofa/simulation/MechanicalVisitor.h>

#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>


namespace sofa
{

namespace component
{

namespace constraintset
{

namespace
{ // helper methods

/// Compute compliance between 2 constraint Jacobians for Vec types
template<int N, typename Real, class VecReal>
inline double UncoupledConstraintCorrection_computeCompliance(
    unsigned int index,
    const sofa::defaulttype::Vec<N, Real>& n1, const sofa::defaulttype::Vec<N, Real>& n2,
    const Real comp0, const VecReal& comp)
{
    return (n1 * n2) * ((index < comp.size()) ? comp[index] : comp0);
}

/// Compute compliance between 2 constraint Jacobians for Rigid types
template<typename Real, class VecReal>
inline double UncoupledConstraintCorrection_computeCompliance(
    unsigned int /*index*/,
    const sofa::defaulttype::RigidDeriv<3, Real>& n1, const sofa::defaulttype::RigidDeriv<3, Real>& n2,
    const Real /*comp0*/, const VecReal& comp)
{
    double w;

    // translation part
    w = (n1.getVCenter() * n2.getVCenter()) * comp[0];
    // rotation part
    w += (n1.getVOrientation()[0] * comp[1] + n1.getVOrientation()[1] * comp[2] + n1.getVOrientation()[2] * comp[3]) * n2.getVOrientation()[0];
    w += (n1.getVOrientation()[0] * comp[2] + n1.getVOrientation()[1] * comp[4] + n1.getVOrientation()[2] * comp[5]) * n2.getVOrientation()[1];
    w += (n1.getVOrientation()[0] * comp[3] + n1.getVOrientation()[1] * comp[5] + n1.getVOrientation()[2] * comp[6]) * n2.getVOrientation()[2];

    return w;
}

/// Compute displacement from constraint force for Vec types
template<int N, typename Real, class VecReal>
inline sofa::defaulttype::Vec<N, Real> UncoupledConstraintCorrection_computeDx(
    unsigned int index,
    const sofa::defaulttype::Vec<N, Real>& f,
    const Real comp0, const VecReal& comp)
{
    return (f) * ((index < comp.size()) ? comp[index] : comp0);
}

/// Compute displacement from constraint force for Rigid types
template<typename Real, class VecReal>
inline sofa::defaulttype::RigidDeriv<3, Real> UncoupledConstraintCorrection_computeDx(
    unsigned int /*index*/,
    const sofa::defaulttype::RigidDeriv<3, Real>& f,
    const Real /*comp0*/, const VecReal& comp)
{
    sofa::defaulttype::RigidDeriv<3, Real> dx;
    // translation part
    dx.getVCenter() = (f.getVCenter()) * comp[0];
    // rotation part
    dx.getVOrientation()[0] = (f.getVOrientation()[0] * comp[1] + f.getVOrientation()[1] * comp[2] + f.getVOrientation()[2] * comp[3]);
    dx.getVOrientation()[1] = (f.getVOrientation()[0] * comp[2] + f.getVOrientation()[1] * comp[4] + f.getVOrientation()[2] * comp[5]);
    dx.getVOrientation()[2] = (f.getVOrientation()[0] * comp[3] + f.getVOrientation()[1] * comp[5] + f.getVOrientation()[2] * comp[6]);

    return dx;
}

}

template<class DataTypes>
UncoupledConstraintCorrection<DataTypes>::UncoupledConstraintCorrection(sofa::core::behavior::MechanicalState<DataTypes> *mm)
    : Inherit(mm)
    , compliance(initData(&compliance, "compliance", "compliance value on each dof"))
    , defaultCompliance(initData(&defaultCompliance, (Real)0.00001, "defaultCompliance", "Default compliance value for new dof or if all should have the same (in which case compliance vector should be empty)"))
    , f_verbose( initData(&f_verbose,false,"verbose","Dump the constraint matrix at each iteration") )
    , d_handleTopologyChange(initData(&d_handleTopologyChange, true, "handleTopologyChange", "Enable support of topological changes for compliance vector (disable if another component takes care of this)"))
    , d_correctionVelocityFactor(initData(&d_correctionVelocityFactor, (Real)1.0, "correctionVelocityFactor", "Factor applied to the constraint forces when correcting the velocities"))
    , d_correctionPositionFactor(initData(&d_correctionPositionFactor, (Real)1.0, "correctionPositionFactor", "Factor applied to the constraint forces when correcting the positions"))
    , d_useOdeSolverIntegrationFactors(initData(&d_useOdeSolverIntegrationFactors, false, "useOdeSolverIntegrationFactors", "Use odeSolver integration factors instead of correctionVelocityFactor and correctionPositionFactor"))
    , m_pOdeSolver(NULL)
{
}

template<class DataTypes>
UncoupledConstraintCorrection<DataTypes>::~UncoupledConstraintCorrection()
{
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::init()
{
    Inherit::init();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (compliance.getValue().size() == 1 && defaultCompliance.isSet() && defaultCompliance.getValue() == compliance.getValue()[0])
    {
        // the same compliance was set in the two data, only keep the default one so that the compliance vector does not need to be maintained
        compliance.setValue(VecReal());
    }

    const VecReal& comp = compliance.getValue();
    if (x.size() != comp.size() && !comp.empty())
    {
        if (!defaultCompliance.isSet() && !comp.empty())
            defaultCompliance.setValue(comp[0]); // set default compliance to first one in case it was given in the compliance vector
        if (comp.size() > 1)
            serr << "Warning compliance size ( " << comp.size() << " is not equal to the size of the mstate (" << x.size() << ")" << sendl;
        Real comp0 = (!comp.empty()) ? comp[0] : defaultCompliance.getValue();
        sout << "Using " << comp0 << " as initial compliance" << sendl;
        VecReal UsedComp;
        for (unsigned int i=0; i<x.size(); i++)
        {
            UsedComp.push_back(comp0);
        }
        // Keeps user specified compliance even if the initial MState size is null.
        if (!UsedComp.empty())
        {
            compliance.setValue(UsedComp);
        }
    }

    this->getContext()->get(m_pOdeSolver);
    if (!m_pOdeSolver)
    {
        if (d_useOdeSolverIntegrationFactors.getValue() == true)
        {
            serr << "Can't find any odeSolver" << sendl;
            d_useOdeSolverIntegrationFactors.setValue(false);
        }
        d_useOdeSolverIntegrationFactors.setReadOnly(true);
    }
}

template <class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::reinit()
{
    Inherit::reinit();
}

template< class DataTypes >
void UncoupledConstraintCorrection< DataTypes >::handleTopologyChange()
{
    using sofa::core::topology::TopologyChange;
    using sofa::core::topology::TopologyChangeType;
    using sofa::core::topology::BaseMeshTopology;

    if (!d_handleTopologyChange.getValue())
        return; // another component takes care of updating compliance vector

    BaseMeshTopology *topology = this->getContext()->getMeshTopology();
    if (!topology)
        return;
    if (defaultCompliance.isSet() && compliance.getValue().empty())
        return; // uniform compliance, no need to update it

    std::list< const TopologyChange * >::const_iterator itBegin = topology->beginChange();
    std::list< const TopologyChange * >::const_iterator itEnd = topology->endChange();

    VecReal& comp = *(compliance.beginEdit());
    const Real comp0 = defaultCompliance.getValue();

    for (std::list< const TopologyChange * >::const_iterator changeIt = itBegin; changeIt != itEnd; ++changeIt)
    {
        const TopologyChangeType changeType = (*changeIt)->getChangeType();

        switch ( changeType )
        {
        case core::topology::POINTSADDED :
        {
            unsigned int nbPoints = (static_cast< const sofa::core::topology::PointsAdded *> (*changeIt))->getNbAddedVertices();

            VecReal addedCompliance;

            for (unsigned int i = 0; i < nbPoints; i++)
            {
                addedCompliance.push_back(comp0);
            }

            comp.insert(comp.end(), addedCompliance.begin(), addedCompliance.end());

            break;
        }
        case core::topology::POINTSREMOVED :
        {
            using sofa::helper::vector;

            const vector< unsigned int > &pts = (static_cast< const sofa::core::topology::PointsRemoved * >(*changeIt))->getArray();

            unsigned int lastIndexVec = comp.size() - 1;

            for (unsigned int i = 0; i < pts.size(); i++)
            {
                comp[pts[i]] = comp[lastIndexVec];
                lastIndexVec--;
            }

            comp.resize(comp.size() - pts.size());

            break;
        }
        default:
            break;
        }
    }

    compliance.endEdit();
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::getComplianceWithConstraintMerge(defaulttype::BaseMatrix* Wmerged, std::vector<int> &constraint_merge)
{
    helper::WriteAccessor<Data<MatrixDeriv> > constraintsData = *this->mstate->write(core::MatrixDerivId::constraintJacobian());
    MatrixDeriv& constraints = constraintsData.wref();

    MatrixDeriv constraintCopy;

    msg_info() << "******\n Constraint before Merge  \n *******" ;

    MatrixDerivRowIterator rowIt = constraints.begin();
    MatrixDerivRowIterator rowItEnd = constraints.end();

    while (rowIt != rowItEnd)
    {
        constraintCopy.writeLine(rowIt.index(), rowIt.row());
        ++rowIt;
    }

    /////////// MERGE OF THE CONSTRAINTS //////////////
    constraints.clear();

    // look for the number of group;
    unsigned int numGroup = 0;
    for (unsigned int cm = 0; cm < constraint_merge.size(); cm++)
    {
        if (constraint_merge[cm] > (int) numGroup)
            numGroup = (unsigned int) constraint_merge[cm];
    }
    numGroup += 1;

   msg_info() << "******\n Constraint after Merge  \n *******" ;

    for (unsigned int group = 0; group < numGroup; group++)
    {
        msg_info() << "constraint[" << group << "] : " ;

        MatrixDerivRowIterator rowCopyIt = constraintCopy.begin();
        MatrixDerivRowIterator rowCopyItEnd = constraintCopy.end();

        while (rowCopyIt != rowCopyItEnd)
        {
            if (constraint_merge[rowCopyIt.index()] == (int)group)
            {
                constraints.addLine(group, rowCopyIt.row());
            }

            ++rowCopyIt;
        }
    }

    //////////// compliance computation call //////////
    this->addComplianceInConstraintSpace(sofa::core::ConstraintParams::defaultInstance(), Wmerged);

    /////////// BACK TO THE INITIAL CONSTRAINT SET//////////////

    constraints.clear();
    msg_info() << "******\n Constraint back to initial values  \n *******" ;

    rowIt = constraintCopy.begin();
    rowItEnd = constraintCopy.end();

    while (rowIt != rowItEnd)
    {
        constraints.writeLine(rowIt.index(), rowIt.row());
        ++rowIt;
    }
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::addComplianceInConstraintSpace(const sofa::core::ConstraintParams * cparams, sofa::defaulttype::BaseMatrix *W)
{
    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();
    VecReal comp = compliance.getValue();
    Real comp0 = defaultCompliance.getValue();
    const bool verbose = f_verbose.getValue();
    const bool useOdeIntegrationFactors = d_useOdeSolverIntegrationFactors.getValue();
    // use the OdeSolver to get the position integration factor
    double factor = 1.0;
    switch (cparams->constOrder())
    {
    case core::ConstraintParams::POS_AND_VEL :
    case core::ConstraintParams::POS :
        factor = useOdeIntegrationFactors ? m_pOdeSolver->getPositionIntegrationFactor() : 1.0;
        break;

    case core::ConstraintParams::ACC :
    case core::ConstraintParams::VEL :
        factor = useOdeIntegrationFactors ? m_pOdeSolver->getVelocityIntegrationFactor() : 1.0;
        break;

    default :
        break;
    }

    comp0 *= Real(factor);
    for(std::size_t i=0;i<comp.size(); ++i)
    {
        comp[i] *= Real(factor);
    }


    for (MatrixDerivRowConstIterator rowIt = constraints.begin(), rowItEnd = constraints.end(); rowIt != rowItEnd; ++rowIt)
    {
        int indexCurRowConst = rowIt.index();
        if (rowIt.row().empty()) continue; // ignore constraints with empty Jacobians

        if (verbose)
        {
            sout << "C[" << indexCurRowConst << "]";
        }

        const MatrixDerivColConstIterator colItBegin = rowIt.begin();
        const MatrixDerivColConstIterator colItEnd = rowIt.end();

        // First the compliance of the constraint with itself
        {
            double w = 0.0;
            
            for (MatrixDerivColConstIterator colIt = colItBegin; colIt != colItEnd; ++colIt)
            {
                unsigned int dof = colIt.index();
                Deriv n = colIt.val();

                if (verbose)
                {
                    sout << " dof[" << dof << "]=" << n;
                }

                //w += (n * n) * (dof < comp.size() ? comp[dof] : comp0);
                w += UncoupledConstraintCorrection_computeCompliance(dof, n, n, comp0, comp);
            }
            
            W->add(indexCurRowConst, indexCurRowConst, w);
        }

        if (verbose)
        {
            sout << sendl;
        }

        // Then the compliance with the remaining constraints
        MatrixDerivRowConstIterator rowIt2 = rowIt;
        ++rowIt2;
        for (; rowIt2 != rowItEnd; ++rowIt2)
        {
            int indexCurColConst = rowIt2.index();
            if (rowIt2.row().empty()) continue; // ignore constraints with empty Jacobians

            // To efficiently compute the compliance between rowIt and rowIt2, we can rely on the
            // fact that the values are sorted on both rows to iterate through them in one pass,
            // with a O(n+m) complexity instead of the brute-force O(n*m) nested loop version.

            double w = 0.0;

            MatrixDerivColConstIterator colIt  = colItBegin;
            MatrixDerivColConstIterator colIt2 = rowIt2.begin();
            const MatrixDerivColConstIterator colIt2End = rowIt2.end();

            while (colIt != colItEnd && colIt2 != colIt2End)
            {
                if (colIt.index() < colIt2.index()) // colIt is behind colIt2
                {
                    ++colIt;
                }
                else if (colIt2.index() < colIt.index()) // colIt2 is behind colIt
                {
                    ++colIt2;
                }
                else // colIt and colIt2 are at the same index
                {
                    unsigned int dof = colIt.index();
                    const Deriv& n1 = colIt.val();
                    const Deriv& n2 = colIt2.val();
                    //w += (n1 * n2) * (dof < comp.size() ? comp[dof] : comp0);
                    w += UncoupledConstraintCorrection_computeCompliance(dof, n1, n2, comp0, comp);
                    ++colIt;
                    ++colIt2;
                }
            }

            if (w != 0.0)
            {
                W->add(indexCurRowConst, indexCurColConst, w);
                W->add(indexCurColConst, indexCurRowConst, w);
            }
        }
    }
}

template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::getComplianceMatrix(defaulttype::BaseMatrix *m) const
{
    const VecReal& comp = compliance.getValue();
    const Real comp0 = defaultCompliance.getValue();
    const unsigned int s = this->mstate->getSize(); // comp.size();
    const unsigned int dimension = Coord::size();

    m->resize(s * dimension, s * dimension); //resize must set to zero the content of the matrix

    for (unsigned int l = 0; l < s; ++l)
    {
        for (unsigned int d = 0; d < dimension; ++d)
            m->set(dimension * l + d, dimension * l + d, (l < comp.size() ? comp[l] : comp0));
    }
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::computeDx(const Data< VecDeriv > &f_d)
{
    const VecDeriv& f = f_d.getValue();

    Data< VecDeriv > &dx_d = *this->mstate->write(core::VecDerivId::dx());
    VecDeriv& dx = *dx_d.beginEdit();

    dx.resize(f.size());
    const VecReal& comp = compliance.getValue();
    const Real comp0 = defaultCompliance.getValue();

    for (unsigned int i = 0; i < dx.size(); i++)
    {
        //dx[i] = f[i] * (i < comp.size() ? comp[i] : comp0);
        dx[i] = UncoupledConstraintCorrection_computeDx(i, f[i], comp0, comp);
    }

    dx_d.endEdit();
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::computeAndApplyMotionCorrection(const core::ConstraintParams *cparams, Data< VecCoord > &x_d, Data< VecDeriv > &v_d, Data< VecDeriv > &f_d, const defaulttype::BaseVector *lambda)
{
    this->addConstraintForceInMotionSpace(f_d, lambda);

    computeDx(f_d);

    VecCoord& x = *x_d.beginEdit();
    VecDeriv& v = *v_d.beginEdit();

    VecDeriv& dx = *(this->mstate->write(core::VecDerivId::dx())->beginEdit());

    const VecCoord& x_free = cparams->readX(this->mstate)->getValue();
    const VecDeriv& v_free = cparams->readV(this->mstate)->getValue();
      
    const bool useOdeIntegrationFactors = d_useOdeSolverIntegrationFactors.getValue();

    const Real xFactor = useOdeIntegrationFactors ? Real(m_pOdeSolver->getPositionIntegrationFactor()) : this->d_correctionPositionFactor.getValue();
    const Real vFactor = useOdeIntegrationFactors ? Real(m_pOdeSolver->getVelocityIntegrationFactor()) : (Real)(this->d_correctionVelocityFactor.getValue() / this->getContext()->getDt());

    for (unsigned int i = 0; i < dx.size(); i++)
    {
        const Deriv dxi = dx[i] * xFactor;
        const Deriv dvi = dx[i] * vFactor;
        x[i] = x_free[i] + dxi;
        v[i] = v_free[i] + dvi;
        dx[i] = dxi;
    }

    x_d.endEdit();
    v_d.endEdit();
    this->mstate->write(core::VecDerivId::dx())->endEdit();
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::computeAndApplyPositionCorrection(const core::ConstraintParams *cparams, Data< VecCoord > &x_d, Data< VecDeriv > &f_d, const defaulttype::BaseVector *lambda)
{
    this->addConstraintForceInMotionSpace(f_d, lambda);

    computeDx(f_d);

    VecCoord& x = *x_d.beginEdit();

    const VecCoord& x_free = cparams->readX(this->mstate)->getValue();

    VecDeriv& dx = *(this->mstate->write(core::VecDerivId::dx())->beginEdit());

    const bool useOdeIntegrationFactors = d_useOdeSolverIntegrationFactors.getValue();

    const Real xFactor = useOdeIntegrationFactors ? Real(m_pOdeSolver->getPositionIntegrationFactor()) : this->d_correctionPositionFactor.getValue();

    for (unsigned int i = 0; i < dx.size(); i++)
    {
        const  Deriv dxi = dx[i] * xFactor;
        x[i] = x_free[i] + dxi;
        dx[i] = dxi;
    }

    x_d.endEdit();
    this->mstate->write(core::VecDerivId::dx())->endEdit();
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::computeAndApplyVelocityCorrection(const core::ConstraintParams *cparams, Data< VecDeriv > &v_d, Data< VecDeriv > &f_d, const defaulttype::BaseVector *lambda)
{
    this->addConstraintForceInMotionSpace(f_d, lambda);

    computeDx(f_d);

    VecDeriv& v = *v_d.beginEdit();

    const VecDeriv& v_free = cparams->readV(this->mstate)->getValue();

    VecDeriv& dx = *(this->mstate->write(core::VecDerivId::dx())->beginEdit());

    const bool useOdeIntegrationFactors = d_useOdeSolverIntegrationFactors.getValue();

    const Real vFactor = useOdeIntegrationFactors ? Real(m_pOdeSolver->getVelocityIntegrationFactor()) : this->d_correctionVelocityFactor.getValue();
    
    for (unsigned int i = 0; i < dx.size(); i++)
    {
        const Deriv dvi = dx[i] * vFactor;
        v[i] = v_free[i] + dvi;
        dx[i] = dvi;
    }

    v_d.endEdit();
    this->mstate->write(core::VecDerivId::dx())->endEdit();
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::applyContactForce(const defaulttype::BaseVector *f)
{
    helper::WriteAccessor<Data<VecDeriv> > forceData = *this->mstate->write(core::VecDerivId::externalForce());
    VecDeriv& force = forceData.wref();
    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();
    const VecReal& comp = compliance.getValue();
    const Real comp0 = defaultCompliance.getValue();

    force.resize((this->mstate->read(core::ConstVecCoordId::position())->getValue()).size());

    MatrixDerivRowConstIterator rowItEnd = constraints.end();

    for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
    {
        double fC1 = f->element(rowIt.index());

        if (fC1 != 0.0)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                force[colIt.index()] += colIt.val() * fC1;
            }
        }
    }


    helper::WriteAccessor<Data<VecDeriv> > dxData = *this->mstate->write(core::VecDerivId::dx());
    VecDeriv& dx = dxData.wref();
    helper::WriteAccessor<Data<VecCoord> > xData = *this->mstate->write(core::VecCoordId::position());
    VecCoord& x = xData.wref();
    helper::WriteAccessor<Data<VecDeriv> > vData = *this->mstate->write(core::VecDerivId::velocity());
    VecDeriv& v = vData.wref();
    const VecDeriv& v_free = this->mstate->read(core::ConstVecDerivId::freeVelocity())->getValue();
    const VecCoord& x_free = this->mstate->read(core::ConstVecCoordId::freePosition())->getValue();

    const bool useOdeIntegrationFactors = d_useOdeSolverIntegrationFactors.getValue();

    const Real xFactor = useOdeIntegrationFactors ? Real(m_pOdeSolver->getPositionIntegrationFactor()) : this->d_correctionPositionFactor.getValue();
    const Real vFactor = useOdeIntegrationFactors ? Real(m_pOdeSolver->getVelocityIntegrationFactor()) : (Real)(this->d_correctionVelocityFactor.getValue() / this->getContext()->getDt());

    // Euler integration... will be done in the "integrator" as soon as it exists !
    dx.resize(v.size());

    for (unsigned int i = 0; i < dx.size(); i++)
    {

        // compliance * force
        //dx[i] = force[i] * (i<comp.size() ? comp[i] : comp0);
        dx[i] = UncoupledConstraintCorrection_computeDx(i, force[i], comp0, comp);

        const Deriv dxi = dx[i] * xFactor;
        const Deriv dvi = dx[i] * vFactor;
        x[i] = x_free[i] + dxi;
        v[i] = v_free[i] + dvi;
        dx[i] = dxi;
    }
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::applyPredictiveConstraintForce(const core::ConstraintParams * /*cparams*/, Data< VecDeriv > &f_d, const defaulttype::BaseVector *lambda)
{
    this->setConstraintForceInMotionSpace(f_d, lambda);
}



template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::resetContactForce()
{
    helper::WriteAccessor<Data<VecDeriv> > forceData = *this->mstate->write(core::VecDerivId::externalForce());
    VecDeriv& force = forceData.wref();

    for (unsigned i = 0; i < force.size(); ++i)
    {
        force[i] = Deriv();
    }
}


///////////////////////  new API for non building the constraint system during solving process //
template<class DataTypes>
bool UncoupledConstraintCorrection<DataTypes>::hasConstraintNumber(int index)
{
    const MatrixDeriv &constraints = this->mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

    return (constraints.readLine(index) != constraints.end());
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::resetForUnbuiltResolution(double * f, std::list<unsigned int>& /*renumbering*/)
{
    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

    constraint_disp.clear();
    constraint_disp.resize(this->mstate->getSize());

    constraint_force.clear();
    constraint_force.resize(this->mstate->getSize());

    constraint_dofs.clear();



    for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != constraints.end(); ++rowIt)
    {
        int indexC = rowIt.index();

        // buf the value of force applied on concerned dof : constraint_force
        // buf a table of indice of involved dof : constraint_dofs
        double fC = f[indexC];

        if (fC != 0.0)
        {
            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != rowIt.end(); ++colIt)
            {
                unsigned int dof = colIt.index();
                constraint_force[dof] += colIt.val() * fC;
                constraint_dofs.push_back(dof);
            }
        }
    }

    // constraint_dofs buff the DOF that are involved with the constraints
    // @TODO: should be sorted first ? (or use a std::set instead of a std::list)
    constraint_dofs.unique();
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::addConstraintDisplacement(double * d, int begin, int end)
{
/// in the Vec1Types and Vec3Types case, compliance is a vector of size mstate->getSize()
/// constraint_force contains the force applied on dof involved with the contact
/// TODO : compute a constraint_disp that is updated each time a new force is provided !

    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

    for (int id = begin; id <= end; id++)
    {
        MatrixDerivRowConstIterator curConstraint = constraints.readLine(id);

        if (curConstraint != constraints.end())
        {
            MatrixDerivColConstIterator colIt = curConstraint.begin();
            MatrixDerivColConstIterator colItEnd = curConstraint.end();

            while (colIt != colItEnd)
            {
                d[id] += colIt.val() * constraint_disp[colIt.index()];

                ++colIt;
            }
        }
    }
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::setConstraintDForce(double * df, int begin, int end, bool update)
{
    /// set a force difference on a set of constraints (between constraint number "begin" and constraint number "end"
    /// if update is false, do nothing
    /// if update is true, it computes the displacements due to this delta of force.
    /// As the contact are uncoupled, a displacement is obtained only on dof involved with the constraints

    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();
    const VecReal& comp = compliance.getValue();
    const Real comp0 = defaultCompliance.getValue();

    if (!update)
        return;

    for (int id = begin; id <= end; id++)
    {

        MatrixDerivRowConstIterator curConstraint = constraints.readLine(id);

        if (curConstraint != constraints.end())
        {
            MatrixDerivColConstIterator colIt = curConstraint.begin();
            MatrixDerivColConstIterator colItEnd = curConstraint.end();

            while (colIt != colItEnd)
            {
                const unsigned int dof = colIt.index();

                constraint_force[dof] += colIt.val() * df[id];

                //Deriv dx =  constraint_force[dof] * (dof < comp.size() ? comp[dof] : comp0);
                Deriv dx = UncoupledConstraintCorrection_computeDx(dof, constraint_force[dof], comp0, comp);

                constraint_disp[dof] = dx;

                ++colIt;
            }
        }
    }
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)
{
    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();
    const VecReal& comp = compliance.getValue();
    const Real comp0 = defaultCompliance.getValue();

    for (int id1 = begin; id1 <= end; id1++)
    {

        MatrixDerivRowConstIterator curConstraint = constraints.readLine(id1);

        if (curConstraint == constraints.end()) continue;

        const MatrixDerivColConstIterator colItBegin = curConstraint.begin();
        const MatrixDerivColConstIterator colItEnd   = curConstraint.end();

        // First the compliance of the constraint with itself
        {
            double w = 0.0;
            
            for (MatrixDerivColConstIterator colIt = colItBegin; colIt != colItEnd; ++colIt)
            {
                unsigned int dof = colIt.index();
                Deriv n = colIt.val();
                //w += (n * n) * (dof < comp.size() ? comp[dof] : comp0);
                w += UncoupledConstraintCorrection_computeCompliance(dof, n, n, comp0, comp);
            }
            
            W->add(id1, id1, w);
        }

        // Then the compliance with the remaining constraints
        for (int id2 = id1+1; id2 <= end; id2++)
        {
            MatrixDerivRowConstIterator curConstraint2 = constraints.readLine(id2);

            if (curConstraint2 == constraints.end()) continue;

            // To efficiently compute the compliance between id1 and id2, we can rely on the
            // fact that the values are sorted on both rows to iterate through them in one pass,
            // with a O(n+m) complexity instead of the brute-force O(n*m) nested loop version.

            double w = 0.0;

            MatrixDerivColConstIterator colIt  = colItBegin;
            MatrixDerivColConstIterator colIt2 = curConstraint2.begin();
            const MatrixDerivColConstIterator colIt2End = curConstraint2.end();

            while (colIt != colItEnd && colIt2 != colIt2End)
            {
                if (colIt.index() < colIt2.index()) // colIt is behind colIt2
                {
                    ++colIt;
                }
                else if (colIt2.index() < colIt.index()) // colIt2 is behind colIt
                {
                    ++colIt2;
                }
                else // colIt and colIt2 are at the same index
                {
                    unsigned int dof = colIt.index();
                    const Deriv& n1 = colIt.val();
                    const Deriv& n2 = colIt2.val();
                    //w += (n1 * n2) * (dof < comp.size() ? comp[dof] : comp0);
                    w += UncoupledConstraintCorrection_computeCompliance(dof, n1, n2, comp0, comp);
                    ++colIt;
                    ++colIt2;
                }
            }

            if (w != 0.0)
            {
                W->add(id1, id2, w);
                W->add(id2, id1, w);
            }
        }
    }
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
