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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPRINGFORCEFIELD_INL

#include <SofaDeformable/SpringForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/config.h>
#include <cassert>
#include <iostream>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(MechanicalState* mstate1, MechanicalState* mstate2, SReal _ks, SReal _kd)
    : Inherit(mstate1, mstate2)
    , ks(initData(&ks,_ks,"stiffness","uniform stiffness for the all springs"))
    , kd(initData(&kd,_kd,"damping","uniform damping for the all springs"))
    , showArrowSize(initData(&showArrowSize,0.01f,"showArrowSize","size of the axis"))
    , drawMode(initData(&drawMode,0,"drawMode","The way springs will be drawn:\n- 0: Line\n- 1:Cylinder\n- 2: Arrow"))
    , springs(initData(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
    , maskInUse(false)
{
}

template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(SReal _ks, SReal _kd)
    : ks(initData(&ks,_ks,"stiffness","uniform stiffness for the all springs"))
    , kd(initData(&kd,_kd,"damping","uniform damping for the all springs"))
    , showArrowSize(initData(&showArrowSize,0.01f,"showArrowSize","size of the axis"))
    , drawMode(initData(&drawMode,0,"drawMode","The way springs will be drawn:\n- 0: Line\n- 1:Cylinder\n- 2: Arrow"))
    , springs(initData(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
    , fileSprings(initData(&fileSprings, "fileSprings", "File describing the springs"))
    , maskInUse(false)
{
    this->addAlias(&fileSprings, "filename");
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
}

template <class DataTypes>
void SpringForceField<DataTypes>::init()
{
    // Load
    if (!fileSprings.getValue().empty())
        load(fileSprings.getFullPath().c_str());
    this->Inherit::init();
}

template<class DataTypes>
void SpringForceField<DataTypes>::addSpringForce(Real& ener, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int /*i*/, const Spring& spring)
{
    int a = spring.m1;
    int b = spring.m2;
    Coord u = p2[b]-p1[a];
    Real d = u.norm();
    if( d<1.0e-4 ) // null length => no force
        return;
    Real inverseLength = 1.0f/d;
    u *= inverseLength;
    Real elongation = (Real)(d - spring.initpos);
    ener += elongation * elongation * spring.ks /2;
    Deriv relativeVelocity = v2[b]-v1[a];
    Real elongationVelocity = dot(u,relativeVelocity);
    Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
    Deriv force = u*forceIntensity;
    f1[a]+=force;
    f2[b]-=force;
}

template<class DataTypes>
void SpringForceField<DataTypes>::addForce(
    const core::MechanicalParams* /* mparams */, DataVecDeriv& data_f1, DataVecDeriv& data_f2,
    const DataVecCoord& data_x1, const DataVecCoord& data_x2,
    const DataVecDeriv& data_v1, const DataVecDeriv& data_v2)
{

    VecDeriv&       f1 = *data_f1.beginEdit();
    const VecCoord& x1 = data_x1.getValue();
    const VecDeriv& v1 = data_v1.getValue();
    VecDeriv&       f2 = *data_f2.beginEdit();
    const VecCoord& x2 = data_x2.getValue();
    const VecDeriv& v2 = data_v2.getValue();


    const helper::vector<Spring>& springs= this->springs.getValue();

    f1.resize(x1.size());
    f2.resize(x2.size());
    this->m_potentialEnergy = 0;
    for (unsigned int i=0; i<this->springs.getValue().size(); i++)
    {
        this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, springs[i]);
    }
    data_f1.endEdit();
    data_f2.endEdit();
}

template<class DataTypes>
void SpringForceField<DataTypes>::addDForce(const core::MechanicalParams*, DataVecDeriv&, DataVecDeriv&, const DataVecDeriv&, const DataVecDeriv& )
{
    serr << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead."<<sendl;
}


template<class DataTypes>
SReal SpringForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& data_x1, const DataVecCoord& data_x2) const
{
    const helper::vector<Spring>& springs= this->springs.getValue();
    const VecCoord& p1 =  data_x1.getValue();
    const VecCoord& p2 =  data_x2.getValue();

    SReal ener = 0;

    for (unsigned int i=0; i<springs.size(); i++)
    {
        int a = springs[i].m1;
        int b = springs[i].m2;
        Coord u = p2[b]-p1[a];
        Real d = u.norm();
        Real elongation = (Real)(d - springs[i].initpos);
        ener += elongation * elongation * springs[i].ks /2;
        //std::cout << "spring energy = " << ener << std::endl;
    }

    return ener;
}


template<class DataTypes>
void SpringForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix *, SReal, unsigned int &)
{
    serr << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead."<<sendl;
}



template<class DataTypes>
void SpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    using namespace sofa::defaulttype;

    if (!((this->mstate1 == this->mstate2)?vparams->displayFlags().getShowForceFields():vparams->displayFlags().getShowInteractionForceFields())) return;
    const VecCoord& p1 =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& p2 =this->mstate2->read(core::ConstVecCoordId::position())->getValue();

    std::vector< Vector3 > points[4];
    bool external = (this->mstate1!=this->mstate2);
    //if (!external)
    //	glColor4f(1,1,1,1);
    const helper::vector<Spring>& springs = this->springs.getValue();
    for (unsigned int i=0; i<springs.size(); i++)
    {
        Real d = (p2[springs[i].m2]-p1[springs[i].m1]).norm();
        Vector3 point2,point1;
        point1 = DataTypes::getCPos(p1[springs[i].m1]);
        point2 = DataTypes::getCPos(p2[springs[i].m2]);

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

    const Vec<4,float> c0(1,0,0,1);
    const Vec<4,float> c1(0,1,0,1);
    const Vec<4,float> c2(1,0.5,0,1);
    const Vec<4,float> c3(0,1,0.5,1);


    if (showArrowSize.getValue()==0 || drawMode.getValue() == 0)
    {
        vparams->drawTool()->drawLines(points[0], 1, c0);
        vparams->drawTool()->drawLines(points[1], 1, c1);
        vparams->drawTool()->drawLines(points[2], 1, c2);
        vparams->drawTool()->drawLines(points[3], 1, c3);
    }
    else if (drawMode.getValue() == 1)
    {
        const unsigned int numLines0=points[0].size()/2;
        const unsigned int numLines1=points[1].size()/2;
        const unsigned int numLines2=points[2].size()/2;
        const unsigned int numLines3=points[3].size()/2;

        for (unsigned int i=0; i<numLines0; ++i) vparams->drawTool()->drawCylinder(points[0][2*i+1], points[0][2*i], showArrowSize.getValue(), c0);
        for (unsigned int i=0; i<numLines1; ++i) vparams->drawTool()->drawCylinder(points[1][2*i+1], points[1][2*i], showArrowSize.getValue(), c1);
        for (unsigned int i=0; i<numLines2; ++i) vparams->drawTool()->drawCylinder(points[2][2*i+1], points[2][2*i], showArrowSize.getValue(), c2);
        for (unsigned int i=0; i<numLines3; ++i) vparams->drawTool()->drawCylinder(points[3][2*i+1], points[3][2*i], showArrowSize.getValue(), c3);

    }
    else if (drawMode.getValue() == 2)
    {
        const unsigned int numLines0=points[0].size()/2;
        const unsigned int numLines1=points[1].size()/2;
        const unsigned int numLines2=points[2].size()/2;
        const unsigned int numLines3=points[3].size()/2;

        for (unsigned int i=0; i<numLines0; ++i) vparams->drawTool()->drawArrow(points[0][2*i+1], points[0][2*i], showArrowSize.getValue(), c0);
        for (unsigned int i=0; i<numLines1; ++i) vparams->drawTool()->drawArrow(points[1][2*i+1], points[1][2*i], showArrowSize.getValue(), c1);
        for (unsigned int i=0; i<numLines2; ++i) vparams->drawTool()->drawArrow(points[2][2*i+1], points[2][2*i], showArrowSize.getValue(), c2);
        for (unsigned int i=0; i<numLines3; ++i) vparams->drawTool()->drawArrow(points[3][2*i+1], points[3][2*i], showArrowSize.getValue(), c3);
    }
    else serr << "No proper drawing mode found!" << sendl;

}

template<class DataTypes>
void SpringForceField<DataTypes>::handleTopologyChange(core::topology::Topology *topo)
{
    if(this->mstate1->getContext()->getTopology() == topo)
    {
        core::topology::BaseMeshTopology*	_topology = topo->toBaseMeshTopology();

        if(_topology != NULL)
        {
            std::list<const core::topology::TopologyChange *>::const_iterator itBegin=_topology->beginChange();
            std::list<const core::topology::TopologyChange *>::const_iterator itEnd=_topology->endChange();

            while( itBegin != itEnd )
            {
                core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

                switch( changeType )
                {
                case core::topology::POINTSREMOVED:
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
        core::topology::BaseMeshTopology*	_topology = topo->toBaseMeshTopology();

        if(_topology != NULL)
        {
            std::list<const core::topology::TopologyChange *>::const_iterator changeIt=_topology->beginChange();
            std::list<const core::topology::TopologyChange *>::const_iterator itEnd=_topology->endChange();

            while( changeIt != itEnd )
            {
                core::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();

                switch( changeType )
                {
                case core::topology::POINTSREMOVED:
                {
                    int nbPoints = _topology->getNbPoints();
                    const sofa::helper::vector<unsigned int>& tab = (static_cast<const sofa::core::topology::PointsRemoved *>(*changeIt))->getArray();

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
void SpringForceField<DataTypes>::updateForceMask()
{
    const helper::vector<Spring>& springs= this->springs.getValue();

    for( unsigned int i=0, iend=springs.size() ; i<iend ; ++i )
    {
        const Spring& s = springs[i];
        this->mstate1->forceMask.insertEntry(s.m1);
        this->mstate2->forceMask.insertEntry(s.m2);
    }
}


template<class DataTypes>
void SpringForceField<DataTypes>::initGnuplot(const std::string path)
{
    if (!this->getName().empty())
    {
        if (m_gnuplotFileEnergy != NULL)
        {
            m_gnuplotFileEnergy->close();
            delete m_gnuplotFileEnergy;
        }
        m_gnuplotFileEnergy = new std::ofstream( (path+this->getName()+"_PotentialEnergy.txt").c_str() );
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::exportGnuplot(SReal time)
{
    if (m_gnuplotFileEnergy!=NULL)
    {
        (*m_gnuplotFileEnergy) << time <<"\t"<< this->m_potentialEnergy << std::endl;
    }
}

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPRINGFORCEFIELD_INL */
