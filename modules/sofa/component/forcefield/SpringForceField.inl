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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_INL

#include <sofa/component/forcefield/SpringForceField.h>
#include <sofa/core/componentmodel/behavior/PairInteractionForceField.inl>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/component/topology/PointSetTopologyChange.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/config.h>
#include <assert.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{




template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(MechanicalState* mstate1, MechanicalState* mstate2, SReal _ks, SReal _kd)
    : Inherit(mstate1, mstate2), maskInUse(false)
    , ks(initData(&ks,_ks,"stiffness","uniform stiffness for the all springs"))
    , kd(initData(&kd,_kd,"damping","uniform damping for the all springs"))
    , springs(initData(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
{
}

template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(SReal _ks, SReal _kd)
    : maskInUse(false)
    , ks(initData(&ks,_ks,"stiffness","uniform stiffness for the all springs"))
    , kd(initData(&kd,_kd,"damping","uniform damping for the all springs"))
    , springs(initData(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
{
}


template<class DataTypes>
void SpringForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    if (arg->getAttribute("filename"))
        this->load(arg->getAttribute("filename"));
    this->Inherit::parse(arg);
}

template <class DataTypes>
class SpringForceField<DataTypes>::Loader : public helper::io::MassSpringLoader
{
public:
    SpringForceField<DataTypes>* dest;
    Loader(SpringForceField<DataTypes>* dest) : dest(dest) {}
    virtual void addSpring(int m1, int m2, SReal ks, SReal kd, SReal initpos)
    {
        helper::vector<Spring>& springs = *dest->springs.beginEdit();
        springs.push_back(Spring(m1,m2,ks,kd,initpos));
        dest->springs.endEdit();
    }
};

template <class DataTypes>
bool SpringForceField<DataTypes>::load(const char *filename)
{
    bool ret = true;
    if (filename && filename[0])
    {
        Loader loader(this);
        ret &= loader.load(filename);
    }
    else ret = false;
    return ret;
}


template <class DataTypes>
void SpringForceField<DataTypes>::reinit()
{
    for (unsigned int i=0; i<springs.getValue().size(); ++i)
    {
        (*springs.beginEdit())[i].ks = (Real) ks.getValue();
        (*springs.beginEdit())[i].kd = (Real) kd.getValue();
    }
    updateMaskStatus();
}

template <class DataTypes>
void SpringForceField<DataTypes>::init()
{
    this->Inherit::init();
    updateMaskStatus();
}

template<class DataTypes>
void SpringForceField<DataTypes>::addSpringForce(SReal& ener, WRefVecDeriv& f1, RRefVecCoord& p1, RRefVecDeriv& v1, WRefVecDeriv& f2, RRefVecCoord& p2, RRefVecDeriv& v2, int /*i*/, const Spring& spring)
{
    int a = spring.m1;
    int b = spring.m2;
    Coord u = p2[b]-p1[a];
    Real d = u.norm();
    Real inverseLength = 1.0f/d;
    if( d>1.0e-4 ) // null length => no force
        return;
    u *= inverseLength;
    Real elongation = (Real)(d - spring.initpos);
    ener += elongation * elongation * spring.ks /2;
    Deriv relativeVelocity = v2[b]-v1[a];
    Real elongationVelocity = dot(u,relativeVelocity);
    Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
    Deriv force = u*forceIntensity;
    f1[a]+=force;
    f2[b]-=force;
    if (this->maskInUse)
    {
        this->mstate1->forceMask.insertEntry(a);
        this->mstate2->forceMask.insertEntry(b);
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addForce(VecDeriv& vf1, VecDeriv& vf2, const VecCoord& vx1, const VecCoord& vx2, const VecDeriv& vv1, const VecDeriv& vv2)
{
    WRefVecDeriv f1 = vf1;
    RRefVecCoord x1 = vx1;
    RRefVecDeriv v1 = vv1;
    WRefVecDeriv f2 = vf2;
    RRefVecCoord x2 = vx2;
    RRefVecDeriv v2 = vv2;

    const helper::vector<Spring>& springs= this->springs.getValue();

    f1.resize(x1.size());
    f2.resize(x2.size());
    m_potentialEnergy = 0;
    for (unsigned int i=0; i<this->springs.getValue().size(); i++)
    {
        this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, springs[i]);
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addDForce(VecDeriv&, VecDeriv&, const VecDeriv&, const VecDeriv&)
{
    serr << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead."<<sendl;
}



template<class DataTypes>
void SpringForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix *, double, unsigned int &)
{
    serr << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead."<<sendl;
}



template<class DataTypes>
void SpringForceField<DataTypes>::draw()
{
    if (!((this->mstate1 == this->mstate2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->mstate1->getX();
    const VecCoord& p2 = *this->mstate2->getX();
    /*        serr<<"SpringForceField<DataTypes>::draw() "<<getName()<<sendl;
            serr<<"SpringForceField<DataTypes>::draw(), p1.size = "<<p1.size()<<sendl;
            serr<<"SpringForceField<DataTypes>::draw(), p1 = "<<p1<<sendl;
            serr<<"SpringForceField<DataTypes>::draw(), p2 = "<<p2<<sendl;*/

    std::vector< Vector3 > points[4];
    bool external = (this->mstate1!=this->mstate2);
    //if (!external)
    //	glColor4f(1,1,1,1);
    const helper::vector<Spring>& springs = this->springs.getValue();

    for (unsigned int i=0; i<springs.size(); i++)
    {
        Real d = (p2[springs[i].m2]-p1[springs[i].m1]).norm();
        Vector3 point1,point2;
        unsigned int sizePoints= (Coord::static_size <=3)?Coord::static_size:3;
        for (unsigned int s=0; s<sizePoints; ++s)
        {
            point1[s] = p1[springs[i].m1][s];
            point2[s] = p2[springs[i].m2][s];
        }
        if (external)
        {
            if (d<springs[i].initpos*0.9999)
            {
                points[0].push_back(point1);
                points[0].push_back(point2);
            }
            else
            {
                points[1].push_back(point1);
                points[1].push_back(point2);
            }
        }
        else
        {
            if (d<springs[i].initpos*0.9999)
            {
                points[2].push_back(point1);
                points[2].push_back(point2);
            }
            else
            {
                points[3].push_back(point1);
                points[3].push_back(point2);
            }
        }
    }


    simulation::getSimulation()->DrawUtility.drawLines(points[0], 1, Vec<4,float>(1,0,0,1));
    simulation::getSimulation()->DrawUtility.drawLines(points[1], 1, Vec<4,float>(0,1,0,1));
    simulation::getSimulation()->DrawUtility.drawLines(points[2], 1, Vec<4,float>(1,0.5,0,1));
    simulation::getSimulation()->DrawUtility.drawLines(points[3], 1, Vec<4,float>(0,1,0.5,1));

}

template<class DataTypes>
void SpringForceField<DataTypes>::handleTopologyChange(core::componentmodel::topology::Topology *topo)
{
    if(this->mstate1->getContext()->getTopology() == topo)
    {
        core::componentmodel::topology::BaseMeshTopology*	_topology = dynamic_cast<core::componentmodel::topology::BaseMeshTopology*> (topo);

        if(_topology != NULL)
        {
            std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=_topology->firstChange();
            std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=_topology->lastChange();

            while( itBegin != itEnd )
            {
                core::componentmodel::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

                switch( changeType )
                {
                case core::componentmodel::topology::POINTSREMOVED:
                {

                    break;
                }

                default:
                    break;
                }; // switch( changeType )

                ++itBegin;
            } // while( changeIt != last; )
        }
    }

    if(this->mstate2->getContext()->getTopology() == topo)
    {
        core::componentmodel::topology::BaseMeshTopology*	_topology = dynamic_cast<core::componentmodel::topology::BaseMeshTopology*> (topo);

        if(_topology != NULL)
        {
            std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator changeIt=_topology->firstChange();
            std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=_topology->lastChange();

            while( changeIt != itEnd )
            {
                core::componentmodel::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();

                switch( changeType )
                {
                case core::componentmodel::topology::POINTSREMOVED:
                {
                    int nbPoints = _topology->getNbPoints();
                    const sofa::helper::vector<unsigned int>& tab = (static_cast<const component::topology::PointsRemoved *>(*changeIt))->getArray();

                    helper::vector<Spring>& springs = *this->springs.beginEdit();
                    // springs.push_back(Spring(m1,m2,ks,kd,initpos));

                    for(unsigned int i=0; i<tab.size(); ++i)
                    {
                        int pntId = tab[i];
                        nbPoints -= 1;

                        for(unsigned int j=0; j<springs.size(); ++j)
                        {
                            Spring& spring = springs[j];
                            if(spring.m2 == pntId)
                            {
                                spring = springs[springs.size() - 1];
                                springs.resize(springs.size() - 1);
                            }

                            if(spring.m2 == nbPoints)
                            {
                                spring.m2 = pntId;
                            }
                        }
                    }

                    this->springs.endEdit();

                    break;
                }

                default:
                    break;
                }; // switch( changeType )

                ++changeIt;
            } // while( changeIt != last; )
        }
    }
}


template <class DataTypes>
void SpringForceField<DataTypes>::updateMaskStatus()
{
    if (this->getMechModel1() == this->getMechModel2()) maskInUse = false;
    else
    {
        if (springs.getValue().size() < 0.5*this->getMechModel1()->getSize() ||
            springs.getValue().size() < 0.5*this->getMechModel2()->getSize() ) maskInUse = true;
        else maskInUse=false;
    }
}

template<class DataTypes>
bool SpringForceField<DataTypes>::useMask()
{
    return maskInUse;
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
