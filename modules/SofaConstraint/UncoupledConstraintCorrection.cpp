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
#define SOFA_COMPONENT_CONSTRAINTSET_UNCOUPLEDCONSTRAINTCORRECTION_CPP

#include "UncoupledConstraintCorrection.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>

#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseMechanics/UniformMass.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;

template<>
SOFA_CONSTRAINT_API UncoupledConstraintCorrection< sofa::defaulttype::Rigid3Types >::UncoupledConstraintCorrection(sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types> *mm)
    : Inherit(mm)
    , compliance(initData(&compliance, "compliance", "Rigid compliance value: 1st value for translations, 6 others for upper-triangular part of symmetric 3x3 rotation compliance matrix"))
{
}

template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::init()
{
    Inherit::init();

    double dt = this->getContext()->getDt();
    const VecReal& comp = compliance.getValue();

    VecReal usedComp;

    if (comp.size() != 7)
    {
        using sofa::component::mass::UniformMass;
        using sofa::defaulttype::Rigid3Types;
        using sofa::defaulttype::Rigid3Mass;
        using sofa::simulation::Node;

        Node *node = dynamic_cast< Node * >(getContext());
        Rigid3Mass massValue;

        //Should use the BaseMatrix API to get the Mass
        //void getElementMass(unsigned int index, defaulttype::BaseMatrix *m)
        if (node != NULL)
        {
            core::behavior::BaseMass *m = node->mass;
            UniformMass< Rigid3Types, Rigid3Mass > *um = dynamic_cast< UniformMass< Rigid3Types, Rigid3Mass >* > (m);

            if (um)
                massValue = um->getMass();
            else
                serr << "WARNING : no mass found" << sendl;
        }
        else
        {
            serr << "\n WARNING : node is not found => massValue could be incorrect in addComplianceInConstraintSpace function" << sendl;
        }

        const double dt2 = dt * dt;

        usedComp.push_back(dt2 / massValue.mass);
        usedComp.push_back(dt2 * massValue.invInertiaMassMatrix[0][0]);
        usedComp.push_back(dt2 * massValue.invInertiaMassMatrix[0][1]);
        usedComp.push_back(dt2 * massValue.invInertiaMassMatrix[0][2]);
        usedComp.push_back(dt2 * massValue.invInertiaMassMatrix[1][1]);
        usedComp.push_back(dt2 * massValue.invInertiaMassMatrix[1][2]);
        usedComp.push_back(dt2 * massValue.invInertiaMassMatrix[2][2]);
        compliance.setValue(usedComp);
    }
    else
    {
        sout << "COMPLIANCE VALUE FOUND" << sendl;
    }
}


#ifndef NEW_VERSION
template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::addComplianceInConstraintSpace(const sofa::core::ConstraintParams * /*cparams*/, defaulttype::BaseMatrix *W)
{
    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::holonomicC())->getValue();

    Deriv weightedNormal;
    Deriv comp_wN;

    const VecReal& usedComp = compliance.getValue();

    MatrixDerivRowConstIterator rowIt = constraints.begin();
    MatrixDerivRowConstIterator rowItEnd = constraints.end();

    while (rowIt != rowItEnd)
    {
        const unsigned int indexCurRowConst = rowIt.index();

        MatrixDerivColConstIterator colIt = rowIt.begin();
        MatrixDerivColConstIterator colItEnd = rowIt.end();

        while (colIt != colItEnd)
        {
            unsigned int dof = colIt.index();
            Deriv n = colIt.val();

            getVCenter(weightedNormal) = getVCenter(n);
            getVOrientation(weightedNormal) = getVOrientation(n);

            getVCenter(comp_wN) = getVCenter(weightedNormal) * usedComp[0];

            const double wn3 = weightedNormal[3];
            const double wn4 = weightedNormal[4];
            const double wn5 = weightedNormal[5];

            comp_wN[3] =  usedComp[1] * wn3 +  usedComp[2] * wn4 +  usedComp[3] * wn5;
            comp_wN[4] =  usedComp[2] * wn3 +  usedComp[4] * wn4 +  usedComp[5] * wn5;
            comp_wN[5] =  usedComp[3] * wn3 +  usedComp[5] * wn4 +  usedComp[6] * wn5;

            MatrixDerivRowConstIterator rowIt2 = rowIt;

            while (rowIt2 != rowItEnd)
            {
                const unsigned int indexCurColConst = rowIt2.index();

                MatrixDerivColConstIterator colIt2 = rowIt2.begin();
                MatrixDerivColConstIterator colIt2End = rowIt2.end();

                while (colIt2 != colIt2End)
                {
                    unsigned int dof2 = colIt2.index();
                    Deriv n2 = colIt2.val();

                    if (dof == dof2)
                    {
                        double w = n2 * comp_wN;

                        W->add(indexCurRowConst, indexCurColConst, w);

                        if (indexCurRowConst != indexCurColConst)
                            W->add(indexCurColConst, indexCurRowConst, w);
                    }

                    ++colIt2;
                }

                ++rowIt2;
            }

            ++colIt;
        }

        ++rowIt;
    }
}

#else

template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::addComplianceInConstraintSpace(const ConstraintParams * /*cparams*/, defaulttype::BaseMatrix *W)
{
    const MatrixDeriv& constraints = *this->mstate->getC();
    const VecReal& usedComp = compliance.getValue();

    Deriv weightedNormal;
    Deriv comp_wN;

    typedef std::list< std::pair< int, Deriv > > CIndicesAndValues;

    helper::vector< CIndicesAndValues > dofsIndexedConstraints;
    const unsigned int numDOFs = this->mstate->getSize();
    dofsIndexedConstraints.resize(numDOFs);

    for (MatrixDerivRowConstIterator rowIt = constraints.begin(), rowItEnd = constraints.end(); rowIt != rowItEnd; ++rowIt)
    {
        int indexCurRowConst = rowIt.index();

        for (MatrixDerivColConstIterator colIt = rowIt.begin(), colItEnd = rowIt.end(); colIt != colItEnd; ++colIt)
        {
            dofsIndexedConstraints[colIt.index()].push_back(std::make_pair(indexCurRowConst, colIt.val()));
        }
    }

    for (MatrixDerivRowConstIterator rowIt = constraints.begin(), rowItEnd = constraints.end(); rowIt != rowItEnd; ++rowIt)
    {
        const unsigned int indexCurRowConst = rowIt.index();

        for (MatrixDerivColConstIterator colIt = rowIt.begin(), colItEnd = rowIt.end(); colIt != colItEnd; ++colIt)
        {
            unsigned int dof = colIt.index();

            CIndicesAndValues &dofsConstraint = dofsIndexedConstraints[dof];

            if (!dofsConstraint.empty())
            {
                Deriv n = colIt.val();

                getVCenter(weightedNormal) = getVCenter(n);
                getVOrientation(weightedNormal) = getVOrientation(n);

                getVCenter(comp_wN) = getVCenter(weightedNormal) * usedComp[0];

                const double wn3 = weightedNormal[3];
                const double wn4 = weightedNormal[4];
                const double wn5 = weightedNormal[5];

                comp_wN[3] =  usedComp[1] * wn3 +  usedComp[2] * wn4 +  usedComp[3] * wn5;
                comp_wN[4] =  usedComp[2] * wn3 +  usedComp[4] * wn4 +  usedComp[5] * wn5;
                comp_wN[5] =  usedComp[3] * wn3 +  usedComp[5] * wn4 +  usedComp[6] * wn5;

                double w = dofsConstraint.front().second * comp_wN;
                W->add(indexCurRowConst, indexCurRowConst, w);
                dofsConstraint.pop_front();

                for (CIndicesAndValues::const_iterator it = dofsConstraint.begin(), itEnd = dofsConstraint.end(); it != itEnd; ++it)
                {
                    w = it->second * comp_wN;
                    W->add(indexCurRowConst, it->first, w);
                    W->add(it->first, indexCurRowConst, w);
                }
            }
        }
    }
}

#endif

template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::getComplianceMatrix(defaulttype::BaseMatrix *m) const
{
    const VecReal& comp = compliance.getValue();
    const unsigned int dimension = defaulttype::DataTypeInfo<Deriv>::size();
    const unsigned int numDofs = comp.size() / 7;

    m->resize(dimension * numDofs, dimension * numDofs);

    /// @todo Optimization
    for (unsigned int d = 0; d < numDofs; ++d)
    {
        const unsigned int d6 = 6 * d;
        const unsigned int d7 = 7 * d;
        const SReal invM = comp[d7];

        m->set(d6, d6, invM);
        m->set(d6 + 1, d6 + 1, invM);
        m->set(d6 + 2, d6 + 2, invM);

        m->set(d6 + 3, d6 + 3, comp[d7 + 1]);

        m->set(d6 + 3, d6 + 4, comp[d7 + 2]);
        m->set(d6 + 4, d6 + 3, comp[d7 + 2]);

        m->set(d6 + 3, d6 + 5, comp[d7 + 3]);
        m->set(d6 + 5, d6 + 3, comp[d7 + 3]);

        m->set(d6 + 4, d6 + 4, comp[d7 + 4]);

        m->set(d6 + 4, d6 + 5, comp[d7 + 5]);
        m->set(d6 + 5, d6 + 4, comp[d7 + 5]);

        m->set(d6 + 5, d6 + 5, comp[d7 + 6]);
    }

    /*
    for (unsigned int l=0;l<s;++l)
    {
        for (unsigned int c=0;c<s;++c)
        {
            if (l==c)
                m->set(l,c,comp[l]);
            else
                m->set(l,c,0);
        }
    }
    */
}


template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::computeDx(const Data< VecDeriv > &f_d)
{
    const VecDeriv& f = f_d.getValue();

    Data< VecDeriv > &dx_d = *this->mstate->write(core::VecDerivId::dx());
    VecDeriv& dx = *dx_d.beginEdit();

    const VecReal& usedComp = compliance.getValue();

    dx.resize(f.size());

    for (unsigned int i = 0; i < dx.size(); i++)
    {
        getVCenter(dx[i]) = getVCenter(f[i]) * usedComp[0];
        dx[i][3] =  usedComp[1] * f[i][3] +  usedComp[2] * f[i][4] +  usedComp[3] * f[i][5];
        dx[i][4] =  usedComp[2] * f[i][3] +  usedComp[4] * f[i][4] +  usedComp[5] * f[i][5];
        dx[i][5] =  usedComp[3] * f[i][3] +  usedComp[5] * f[i][4] +  usedComp[6] * f[i][5];
    }

    dx_d.endEdit();
}


template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::applyContactForce(const defaulttype::BaseVector *f)
{
    helper::WriteAccessor<Data<VecDeriv> > forceData = *this->mstate->write(core::VecDerivId::externalForce());
    VecDeriv& force = forceData.wref();
    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::holonomicC())->getValue();

    unsigned int dof;
    Deriv weightedNormal;

    const VecReal& usedComp = compliance.getValue();

    force.resize((this->mstate->read(core::ConstVecCoordId::position())->getValue()).size());

    MatrixDerivRowConstIterator rowIt = constraints.begin();
    MatrixDerivRowConstIterator rowItEnd = constraints.end();

    while (rowIt != rowItEnd)
    {
        const double fC1 = f->element(rowIt.index());

        if (fC1 != 0.0)
        {
            MatrixDerivColConstIterator colIt = rowIt.begin();
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            while (colIt != colItEnd)
            {
                dof = colIt.index();
                weightedNormal = colIt.val();

                getVCenter(force[dof]) += getVCenter(weightedNormal) * fC1;
                getVOrientation(force[dof]) += getVOrientation(weightedNormal) * fC1;

                ++colIt;
            }
        }

        ++rowIt;
    }

    helper::WriteAccessor<Data<VecDeriv> > dxData = *this->mstate->write(core::VecDerivId::dx());
    VecDeriv& dx = dxData.wref();
    helper::WriteAccessor<Data<VecCoord> > xData = *this->mstate->write(core::VecCoordId::position());
    VecCoord& x = xData.wref();
    helper::WriteAccessor<Data<VecDeriv> > vData = *this->mstate->write(core::VecDerivId::velocity());
    VecDeriv& v = vData.wref();

    const VecDeriv& v_free = this->mstate->read(core::ConstVecDerivId::freeVelocity())->getValue();
    const VecCoord& x_free = this->mstate->read(core::ConstVecCoordId::freePosition())->getValue();

    const double dt = this->getContext()->getDt();

    // Euler integration... will be done in the "integrator" as soon as it exists !
    dx.resize(v.size());

    for (unsigned int i = 0; i < dx.size(); i++)
    {
        x[i] = x_free[i];
        v[i] = v_free[i];

        // compliance * force
        getVCenter(dx[i]) = getVCenter(force[i]) * usedComp[0];
        dx[i][3] =  usedComp[1] * force[i][3] +  usedComp[2] * force[i][4] +  usedComp[3] * force[i][5];
        dx[i][4] =  usedComp[2] * force[i][3] +  usedComp[4] * force[i][4] +  usedComp[5] * force[i][5];
        dx[i][5] =  usedComp[3] * force[i][3] +  usedComp[5] * force[i][4] +  usedComp[6] * force[i][5];
        dx[i] *= (1.0 / dt);
        v[i] += dx[i];
        dx[i] *= dt;
        x[i] += dx[i];
    }
}


template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::setConstraintDForce(double * df, int begin, int end, bool update)
{
    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::holonomicC())->getValue();
    const VecReal& usedComp = compliance.getValue();

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
                Deriv n = colIt.val();
                unsigned int dof = colIt.index();

                constraint_force[dof] += n * df[id];

                Deriv dx;
                getVCenter(dx) = getVCenter(constraint_force[dof]) * usedComp[0];

                defaulttype::Vec3d wrench = getVOrientation(constraint_force[dof]);
                getVOrientation(dx)[0] = usedComp[1] * wrench[0] + usedComp[2] * wrench[1] + usedComp[3] * wrench[2];
                getVOrientation(dx)[1] = usedComp[2] * wrench[0] + usedComp[4] * wrench[1] + usedComp[5] * wrench[2];
                getVOrientation(dx)[2] = usedComp[3] * wrench[0] + usedComp[5] * wrench[1] + usedComp[6] * wrench[2];

                constraint_disp[dof] = dx;

                ++colIt;
            }
        }
    }
}


///////////////////// ATTENTION : passer un indice début - fin (comme pour force et déplacement) pour calculer le block complet
///////////////////// et pas uniquement la diagonale.
template<>
SOFA_CONSTRAINT_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)
{
    const MatrixDeriv& constraints = this->mstate->read(core::ConstMatrixDerivId::holonomicC())->getValue();
    const VecReal& usedComp = compliance.getValue();

    msg_info()<<"getBlockDiagonalCompliance called for lines and columns "<< begin<< " to "<< end ;

    Deriv weightedNormal, C_n;

    for (int id1 = begin; id1 <= end; id1++)
    {

        MatrixDerivRowConstIterator curConstraint = constraints.readLine(id1);

        if (curConstraint != constraints.end())
        {
            MatrixDerivColConstIterator colIt = curConstraint.begin();
            MatrixDerivColConstIterator colItEnd = curConstraint.end();

            while (colIt != colItEnd)
            {
                weightedNormal = colIt.val();
                unsigned int dof1 = colIt.index();

                getVCenter(C_n) = getVCenter(weightedNormal) * usedComp[0];
                defaulttype::Vec3d wrench = getVOrientation(weightedNormal) ;

                getVOrientation(C_n)[0] = usedComp[1] * wrench[0] + usedComp[2] * wrench[1] + usedComp[3] * wrench[2];
                getVOrientation(C_n)[1] = usedComp[2] * wrench[0] + usedComp[4] * wrench[1] + usedComp[5] * wrench[2];
                getVOrientation(C_n)[2] = usedComp[3] * wrench[0] + usedComp[5] * wrench[1] + usedComp[6] * wrench[2];

                for (int id2 = id1; id2 <= end; id2++)
                {

                    MatrixDerivRowConstIterator curConstraint2 = constraints.readLine(id2);

                    if (curConstraint2 != constraints.end())
                    {
                        MatrixDerivColConstIterator colIt2 = curConstraint2.begin();
                        MatrixDerivColConstIterator colItEnd2 = curConstraint2.end();

                        while (colIt2 != colItEnd2)
                        {
                            unsigned int dof2 = colIt2.index();

                            if (dof1 == dof2)
                            {
                                Deriv n2 = colIt2.val();
                                double w = n2 * C_n;

                                W->add(id1, id2, w);

                                if (id1 != id2)
                                    W->add(id2, id1, w);
                            }

                            ++colIt2;
                        }
                    }
                }

                ++colIt;
            }
        }
    }
}


SOFA_DECL_CLASS(UncoupledConstraintCorrection)

int UncoupledConstraintCorrectionClass = core::RegisterObject("Component computing contact forces within a simulated body using the compliance method.")
#ifndef SOFA_FLOAT
        .add< UncoupledConstraintCorrection< Vec1dTypes > >()
        .add< UncoupledConstraintCorrection< Vec2dTypes > >()
        .add< UncoupledConstraintCorrection< Vec3dTypes > >()
        .add< UncoupledConstraintCorrection< Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< UncoupledConstraintCorrection< Vec1fTypes > >()
        .add< UncoupledConstraintCorrection< Vec2fTypes > >()
        .add< UncoupledConstraintCorrection< Vec3fTypes > >()
        .add< UncoupledConstraintCorrection< Rigid3fTypes > >()
        //TODO(dmarchal) There is no Rigid3fTypes template specizaliation while there is one for Rigid3d...
        //this look sucipicious.

#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec1dTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec2dTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec3dTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec1fTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec2fTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Vec3fTypes >;
template class SOFA_CONSTRAINT_API UncoupledConstraintCorrection< Rigid3fTypes >;
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa
