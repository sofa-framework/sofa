/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_INL

#include <SofaConstraint/BilateralInteractionConstraint.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>

#include <algorithm> // for std::min

namespace sofa
{

namespace component
{

namespace constraintset
{

namespace bilateralinteractionconstraint
{
using sofa::core::objectmodel::KeypressedEvent ;
using sofa::core::objectmodel::Event ;
using sofa::helper::WriteAccessor ;
using sofa::defaulttype::Vec;

//TODO(dmarchal): isn't ths function already defined somewhere in SOFA ?
template<typename T>
inline double sign(T &toto)
{
    if (toto<0.0)
        return -1.0;
    return 1.0;
}

inline defaulttype::Quat qDiff(defaulttype::Quat a, const defaulttype::Quat& b)
{
    if (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]<0)
    {
        a[0] = -a[0];
        a[1] = -a[1];
        a[2] = -a[2];
        a[3] = -a[3];
    }
    defaulttype::Quat q = b.inverse() * a;
    return q;
}

template<class DataTypes>
BilateralInteractionConstraint<DataTypes>::BilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2)
    : Inherit(object1, object2)
    , m1(initData(&m1, "first_point","index of the constraint on the first model"))
    , m2(initData(&m2, "second_point","index of the constraint on the second model"))
    , restVector(initData(&restVector, "rest_vector","Relative position to maintain between attached points (optional)"))

    , d_numericalTolerance(initData(&d_numericalTolerance, 0.0001, "numericalTolerance",
                                    "a real value specifying the tolerance during the constraint solving. (optional, default=0.0001)") )

    //TODO(dmarchal): Such kind of behavior shouldn't be implemented in the component but externalized in a second component or in a python script controlling the scene.
    , activateAtIteration( initData(&activateAtIteration, 0, "activateAtIteration", "activate constraint at specified interation (0=disable)"))

    //TODO(dmarchal): what do TEST means in the following ? should it be renamed (EXPERIMENTAL FEATURE) and when those Experimental feature will become official feature ?
    , merge(initData(&merge,false, "merge", "TEST: merge the bilateral constraints in a unique constraint"))
    , derivative(initData(&derivative,false, "derivative", "TEST: derivative"))
    , keepOrientDiff(initData(&keepOrientDiff,false, "keepOrientationDifference", "keep the initial difference in orientation (only for rigids)"))

    , activated(true), iteration(0)

    //0.0001) ;


{
    this->f_listening.setValue(true);
}

template<class DataTypes>
BilateralInteractionConstraint<DataTypes>::BilateralInteractionConstraint(MechanicalState* object)
    : Inherit(object, object)
    , m1(initData(&m1, "first_point","index of the constraint on the first model"))
    , m2(initData(&m2, "second_point","index of the constraint on the second model"))
    , restVector(initData(&restVector, "rest_vector","Relative position to maintain between attached points (optional)"))
    , d_numericalTolerance(initData(&d_numericalTolerance, 0.0001, "numericalTolerance", "a real value specifying the tolerance during the constraint solving. (default=0.0001") )

    , activateAtIteration( initData(&activateAtIteration, 0, "activateAtIteration", "activate constraint at specified interation (0 = always enabled, -1=disabled)"))

    , merge(initData(&merge,false, "merge", "TEST: merge the bilateral constraints in a unique constraint"))
    , derivative(initData(&derivative,false, "derivative", "TEST: derivative"))
    , keepOrientDiff(initData(&keepOrientDiff,false, "keepOrientationDifference", "keep the initial difference in orientation (only for rigids)"))
    , activated(true), iteration(0)
{
    this->f_listening.setValue(true);
}

template<class DataTypes>
BilateralInteractionConstraint<DataTypes>::BilateralInteractionConstraint()
    : m1(initData(&m1, "first_point","index of the constraint on the first model"))
    , m2(initData(&m2, "second_point","index of the constraint on the second model"))
    , restVector(initData(&restVector, "rest_vector","Relative position to maintain between attached points (optional)"))
    , d_numericalTolerance(initData(&d_numericalTolerance, 0.0001, "numericalTolerance", "a real value specifying the tolerance during the constraint solving. (default=0.0001") )

    , activateAtIteration( initData(&activateAtIteration, 0, "activateAtIteration", "activate constraint at specified interation (0 = always enabled, -1=disabled)"))

    , merge(initData(&merge,false, "merge", "TEST: merge the bilateral constraints in a unique constraint"))
    , derivative(initData(&derivative,false, "derivative", "TEST: derivative"))
    , keepOrientDiff(initData(&keepOrientDiff,false, "keepOrientationDifference", "keep the initial difference in orientation (only for rigids)"))
    , activated(true), iteration(0)
{
    this->f_listening.setValue(true);
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::unspecializedInit()
{
    /// Do general check of validity for inputs
    Inherit1::init();

    /// Using assert means that the previous lines have check that there is two valid mechanical state.
    assert(this->mstate1);
    assert(this->mstate2);

    prevForces.clear();
    iteration = 0;
    activated = (activateAtIteration.getValue() >= 0 && activateAtIteration.getValue() <= iteration);
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::init()
{
    unspecializedInit();
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::reinit()
{
    prevForces.clear();
    activated = (activateAtIteration.getValue() >= 0 && activateAtIteration.getValue() <= iteration);
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::reset(){
    init();
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::buildConstraintMatrix(const ConstraintParams*, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &constraintId
                                                                      , const DataVecCoord &/*x1*/, const DataVecCoord &/*x2*/)
{
    if (!activated)
        return;

    unsigned minp = std::min(m1.getValue().size(), m2.getValue().size());
    if (minp == 0)
        return;

    const helper::vector<int> &m1Indices = m1.getValue();
    const helper::vector<int> &m2Indices = m2.getValue();

    MatrixDeriv &c1 = *c1_d.beginEdit();
    MatrixDeriv &c2 = *c2_d.beginEdit();

    cid.resize(minp);

    const VecDeriv& restVector = this->restVector.getValue();

    if (!merge.getValue())
    {
        for (unsigned pid=0; pid<minp; pid++)
        {
            int tm1 = m1Indices[pid];
            int tm2 = m2Indices[pid];

            const defaulttype::Vec<3, Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);

            cid[pid] = constraintId;
            constraintId += 3;

            MatrixDerivRowIterator c1_it = c1.writeLine(cid[pid]);
            c1_it.addCol(tm1, -cx);

            MatrixDerivRowIterator c2_it = c2.writeLine(cid[pid]);
            c2_it.addCol(tm2, cx);

            c1_it = c1.writeLine(cid[pid] + 1);
            c1_it.setCol(tm1, -cy);

            c2_it = c2.writeLine(cid[pid] + 1);
            c2_it.setCol(tm2, cy);

            c1_it = c1.writeLine(cid[pid] + 2);
            c1_it.setCol(tm1, -cz);

            c2_it = c2.writeLine(cid[pid] + 2);
            c2_it.setCol(tm2, cz);
        }
    }
    else
    {
        this->m_constraintIndex.setValue(constraintId);


        ///////////////// grouped constraints ///////////////
        dfree_square_total.clear();

        const DataVecCoord &d_x1 = *this->mstate1->read(ConstVecCoordId::position());
        const DataVecCoord &d_x2 = *this->mstate2->read(ConstVecCoordId::position());

        const VecCoord &x1 = d_x1.getValue();
        const VecCoord &x2 = d_x2.getValue();

        for (unsigned pid=0; pid<minp; pid++)
        {
            int tm1 = m1Indices[pid];
            int tm2 = m2Indices[pid];

            Deriv dfree_loc = x2[tm2] - x1[tm1];

            if (pid < restVector.size())
                dfree_loc -= restVector[pid];

            dfree_square_total[0]+= dfree_loc[0]*dfree_loc[0];
            dfree_square_total[1]+= dfree_loc[1]*dfree_loc[1];
            dfree_square_total[2]+= dfree_loc[2]*dfree_loc[2];
        }

        for (unsigned int i=0; i<3; i++)
        {
            if (dfree_square_total[i]>1.0e-15)
            {
                dfree_square_total[i] = sqrt(dfree_square_total[i]);
                squareXYZ[i]=derivative.getValue();
            }
            else
                squareXYZ[i]=false;
        }

        dfree.resize(minp);

        const DataVecCoord &d_x1free = *this->mstate1->read(ConstVecCoordId::freePosition());
        const DataVecCoord &d_x2free = *this->mstate2->read(ConstVecCoordId::freePosition());

        const VecCoord &x1free = d_x1free.getValue();
        const VecCoord &x2free = d_x2free.getValue();

        for (unsigned pid=0; pid<minp; pid++)
        {
            int tm1 = m1Indices[pid];
            int tm2 = m2Indices[pid];

            Deriv d_loc = x2[tm2] - x1[tm1];
            Deriv dfree_loc = x2free[tm2] - x1free[tm1];

            if (pid < restVector.size())
            {
                d_loc -= restVector[pid];
                dfree_loc -= restVector[pid];
            }
            dfree[pid] = dfree_loc;

            const defaulttype::Vec<3, Real> cx(1.0,0,0), cy(0,1.0,0), cz(0,0,1.0);

            cid[pid] = constraintId;


            // if not grouped constraint
            // constraintId += 3;

            // contribution along x axis
            MatrixDerivRowIterator c1_it = c1.writeLine(cid[pid]);
            MatrixDerivRowIterator c2_it = c2.writeLine(cid[pid]);
            if(squareXYZ[0])
            {
                c1_it.addCol(tm1, -cx*dfree_loc[0]*2.0);
                c2_it.addCol(tm2, cx*dfree_loc[0]*2.0);
            }
            else
            {
                c1_it.addCol(tm1, -cx*sign(dfree_loc[0]) );
                c2_it.addCol(tm2, cx*sign(dfree_loc[0]));
            }


            // contribution along y axis
            c1_it = c1.writeLine(cid[pid] + 1);
            c2_it = c2.writeLine(cid[pid] + 1);
            if(squareXYZ[1])
            {

                c1_it.addCol(tm1, -cy*dfree_loc[1]*2.0);
                c2_it.addCol(tm2, cy*dfree_loc[1]*2.0);
            }
            else
            {
                c1_it.addCol(tm1, -cy*sign(dfree_loc[1]));
                c2_it.addCol(tm2, cy*sign(dfree_loc[1]));
            }

            // contribution along z axis
            c1_it = c1.writeLine(cid[pid] + 2);
            c2_it = c2.writeLine(cid[pid] + 2);
            if(squareXYZ[2])
            {
                c1_it.addCol(tm1, -cz*dfree_loc[2]*2.0);
                c2_it.addCol(tm2, cz*dfree_loc[2]*2.0);
            }
            else
            {
                c1_it.addCol(tm1, -cz*sign(dfree_loc[2]));
                c2_it.addCol(tm2, cz*sign(dfree_loc[2]));
            }
        }

        // if grouped constraint
        constraintId += 3;
    }

    c1_d.endEdit();
    c2_d.endEdit();
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintViolation(const ConstraintParams* cParams,
                                                                       BaseVector *v,
                                                                       const DataVecCoord &d_x1, const DataVecCoord &d_x2
                                                                       , const DataVecDeriv & d_v1, const DataVecDeriv & d_v2)
{
    if (!activated) return;

    const helper::vector<int> &m1Indices = m1.getValue();
    const helper::vector<int> &m2Indices = m2.getValue();

    unsigned minp = std::min(m1Indices.size(), m2Indices.size());

    const VecDeriv& restVector = this->restVector.getValue();

    if (cParams->constOrder() == ConstraintParams::VEL)
    {
        getVelocityViolation(v, d_x1, d_x2, d_v1, d_v2);
        return;
    }

    const VecCoord &x1 = d_x1.getValue();
    const VecCoord &x2 = d_x2.getValue();

    if (!merge.getValue())
    {
        dfree.resize(minp);

        for (unsigned pid=0; pid<minp; pid++)
        {
            dfree[pid] = x2[m2Indices[pid]] - x1[m1Indices[pid]];

            if (pid < restVector.size())
                dfree[pid] -= restVector[pid];

            v->set(cid[pid]  , dfree[pid][0]);
            v->set(cid[pid]+1, dfree[pid][1]);
            v->set(cid[pid]+2, dfree[pid][2]);
        }
    }
    else
    {
        for (unsigned pid=0; pid<minp; pid++)
        {
            dfree[pid] = x2[m2Indices[pid]] - x1[m1Indices[pid]];

            if (pid < restVector.size())
                dfree[pid] -= restVector[pid];

            for (unsigned int i=0; i<3; i++)
            {
                if(squareXYZ[i])
                    v->add(cid[pid]+i  , dfree[pid][i]*dfree[pid][i]);
                else
                {

                    v->add(cid[pid]+i  , dfree[pid][i]*sign(dfree[pid][i] ) );
                }
            }

        }
    }
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getVelocityViolation(BaseVector *v,
                                                                     const DataVecCoord &d_x1,
                                                                     const DataVecCoord &d_x2,
                                                                     const DataVecDeriv &d_v1,
                                                                     const DataVecDeriv &d_v2)
{
    std::cout<<"getVelocityViolation called "<<std::endl;

    const helper::vector<int> &m1Indices = m1.getValue();
    const helper::vector<int> &m2Indices = m2.getValue();

    const VecCoord &x1 = d_x1.getValue();
    const VecCoord &x2 = d_x2.getValue();
    const VecCoord &v1 = d_v1.getValue();
    const VecCoord &v2 = d_v2.getValue();

    unsigned minp = std::min(m1Indices.size(), m2Indices.size());
    const VecDeriv& restVector = this->restVector.getValue();
    std::vector<Deriv> dPrimefree;

    if (!merge.getValue())
    {
        dPrimefree.resize(minp);

        for (unsigned pid=0; pid<minp; pid++)
        {
            dPrimefree[pid] = v2[m2Indices[pid]] - v1[m1Indices[pid]];
            if (pid < restVector.size())
                dPrimefree[pid] -= restVector[pid];

            v->set(cid[pid]  , dPrimefree[pid][0]);
            v->set(cid[pid]+1, dPrimefree[pid][1]);
            v->set(cid[pid]+2, dPrimefree[pid][2]);
        }
    }
    else
    {

        dPrimefree.resize(minp);
        dfree.resize(minp);

        for (unsigned pid=0; pid<minp; pid++)
        {
            dPrimefree[pid] = v2[m2Indices[pid]] - v1[m1Indices[pid]];
            dfree[pid] = x2[m2Indices[pid]] - x1[m1Indices[pid]];

            if (pid < restVector.size())
            {
                dPrimefree[pid] -= restVector[pid];
                dfree[pid] -= restVector[pid];
            }

            std::cout<<" x2 : "<<x2[m2Indices[pid]]<<" - x1 :"<<x1[m1Indices[pid]]<<" = "<<dfree[pid]<<std::endl;
            std::cout<<" v2 : "<<v2[m2Indices[pid]]<<" - v1 :"<<v1[m1Indices[pid]]<<" = "<<dPrimefree[pid]<<std::endl;

            for (unsigned int i=0; i<3; i++)
            {
                if(squareXYZ[i])
                {
                    //std::cout<<" vel viol:"<<2*dPrimefree[pid][i]*dfree[pid][i]<<std::endl;
                    v->add(cid[pid]+i  , 2*dPrimefree[pid][i]*dfree[pid][i]);
                }
                else
                {
                    //std::cout<<" vel viol:"<<dPrimefree[pid][i]*sign(dfree[pid][i] )<<std::endl;
                    v->add(cid[pid]+i  , dPrimefree[pid][i]*sign(dfree[pid][i] ) );
                }
            }

        }
    }
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintResolution(const ConstraintParams* cParams,
                                                                        std::vector<ConstraintResolution*>& resTab,
                                                                        unsigned int& offset)
{
    SOFA_UNUSED(cParams);
    unsigned minp=std::min(m1.getValue().size(),m2.getValue().size());

    if (!merge.getValue())
    {
        prevForces.resize(minp);
        for (unsigned pid=0; pid<minp; pid++)
        {
            resTab[offset] = new BilateralConstraintResolution3Dof(&prevForces[pid]);
            offset += 3;
        }
    }
    else
    {
        prevForces.resize(1);
        resTab[offset] = new BilateralConstraintResolution3Dof(&prevForces[0]);
        offset +=3;
    }
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::addContact(Deriv /*norm*/, Coord P, Coord Q,
                                                           Real /*contactDistance*/, int m1, int m2,
                                                           Coord /*Pfree*/, Coord /*Qfree*/,
                                                           long /*id*/, PersistentID /*localid*/)
{
    WriteAccessor<Data<helper::vector<int> > > wm1 = this->m1;
    WriteAccessor<Data<helper::vector<int> > > wm2 = this->m2;
    WriteAccessor<Data<VecDeriv > > wrest = this->restVector;
    wm1.push_back(m1);
    wm2.push_back(m2);
    wrest.push_back(Q-P);
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::addContact(Deriv norm, Coord P, Coord Q, Real
                                                           contactDistance, int m1, int m2,
                                                           long id, PersistentID localid)
{
   addContact(norm, P, Q, contactDistance, m1, m2,
               this->getMState2()->read(ConstVecCoordId::freePosition())->getValue()[m2],
               this->getMState1()->read(ConstVecCoordId::freePosition())->getValue()[m1],
               id, localid);
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::addContact(Deriv norm, Real contactDistance,
                                                           int m1, int m2, long id, PersistentID localid)
{
    addContact(norm,
               this->getMState2()->read(ConstVecCoordId::position())->getValue()[m2],
               this->getMState1()->read(ConstVecCoordId::position())->getValue()[m1],
               contactDistance, m1, m2,
               this->getMState2()->read(ConstVecCoordId::freePosition())->getValue()[m2],
               this->getMState1()->read(ConstVecCoordId::freePosition())->getValue()[m1],
               id, localid);
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::clear(int reserve)
{
    WriteAccessor<Data<helper::vector<int> > > wm1 = this->m1;
    WriteAccessor<Data<helper::vector<int> > > wm2 = this->m2;
    WriteAccessor<Data<VecDeriv > > wrest = this->restVector;
    wm1.clear();
    wm2.clear();
    wrest.clear();
    if (reserve)
    {
        wm1.reserve(reserve);
        wm2.reserve(reserve);
        wrest.reserve(reserve);
    }
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);
    glPointSize(10);
    if (activated)
        glColor4f(1,0,1,1);
    else
        glColor4f(0,1,0,1);
    glBegin(GL_POINTS);

    unsigned minp = std::min(m1.getValue().size(),m2.getValue().size());
    for (unsigned i=0; i<minp; i++)
    {
        helper::gl::glVertexT(this->mstate1->read(ConstVecCoordId::position())->getValue()[m1.getValue()[i]]);
        helper::gl::glVertexT(this->mstate2->read(ConstVecCoordId::position())->getValue()[m2.getValue()[i]]);
    }
    glEnd();
    glPointSize(1);
#endif /* SOFA_NO_OPENGL */
}

//TODO(dmarchal): implementing keyboard interaction behavior directly in a component is not a valid
//design for a component. Interaction should be defered to an independent Component implemented in the SofaInteraction
//a second possibility is to implement this behavir using script.
template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::handleEvent(Event *event)
{
    if (KeypressedEvent::checkEventType(event))
    {
        KeypressedEvent *ev = static_cast<KeypressedEvent *>(event);
        switch(ev->getKey())
        {

        case 'A':
        case 'a':
            msg_info() << "Activating constraint" ;
            activated = true;
            break;
        }
    }


    if (simulation::AnimateEndEvent::checkEventType(event) )
    {
        ++iteration;
        if (!activated && activateAtIteration.getValue() >= 0 && activateAtIteration.getValue() <= iteration)
        {
            msg_info() << "Activating constraint" ;
            activated = true;
        }
    }
}


} // namespace bilateralinteractionconstraint

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_INL
