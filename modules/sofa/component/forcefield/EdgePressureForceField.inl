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
#include <sofa/component/forcefield/EdgePressureForceField.h>
#include <sofa/component/topology/EdgeSubsetData.inl>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <set>

#ifdef _WIN32
#include <windows.h>
#endif

// #define DEBUG_TRIANGLEFEM

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::topology;





template <class DataTypes> EdgePressureForceField<DataTypes>::~EdgePressureForceField()
{
}
// Handle topological changes
template <class DataTypes> void  EdgePressureForceField<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=_topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=_topology->lastChange();


    edgePressureMap.handleTopologyEvents(itBegin,itEnd,_topology->getNbEdges());

}
template <class DataTypes> void EdgePressureForceField<DataTypes>::init()
{
    //serr << "initializing EdgePressureForceField" << sendl;
    this->core::componentmodel::behavior::ForceField<DataTypes>::init();

    _topology = this->getContext()->getMeshTopology();
    this->getContext()->get(edgeGeo);

    assert(edgeGeo!=0);

    if (edgeGeo==NULL)
    {
        serr << "ERROR(EdgePressureForceField): object must have an EdgeSetTopology."<<sendl;
        return;
    }

    if (dmin.getValue()!=dmax.getValue())
    {
        selectEdgesAlongPlane();
    }
    if (edgeList.getValue().length()>0)
    {
        selectEdgesFromString();
    }

    initEdgeInformation();

}


template <class DataTypes>
void EdgePressureForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& /*x*/, const VecDeriv& /*v*/)
{
    Deriv force;

    typename topology::EdgeSubsetData<EdgePressureInformation>::iterator it;

    for(it=edgePressureMap.begin(); it!=edgePressureMap.end(); it++ )
    {
        force=(*it).second.force/2;
        f[_topology->getEdge((*it).first)[0]]+=force;
        f[_topology->getEdge((*it).first)[1]]+=force;

    }
}

template <class DataTypes>
double EdgePressureForceField<DataTypes>::getPotentialEnergy(const VecCoord& /*x*/)
{
    serr<<"EdgePressureForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}

template<class DataTypes>
void EdgePressureForceField<DataTypes>::initEdgeInformation()
{
    typename topology::EdgeSubsetData<EdgePressureInformation>::iterator it;

    for(it=edgePressureMap.begin(); it!=edgePressureMap.end(); it++ )
    {
        (*it).second.length=edgeGeo->computeRestEdgeLength((*it).first);
        (*it).second.force=pressure.getValue()*(*it).second.length;
    }
}


template<class DataTypes>
void EdgePressureForceField<DataTypes>::updateEdgeInformation()
{
    typename topology::EdgeSubsetData<EdgePressureInformation>::iterator it;

    for(it=edgePressureMap.begin(); it!=edgePressureMap.end(); it++ )
    {
        (*it).second.force=pressure.getValue()*((*it).second.length);
    }
}


template <class DataTypes>
void EdgePressureForceField<DataTypes>::selectEdgesAlongPlane()
{
    const VecCoord& x = *this->mstate->getX0();
    std::vector<bool> vArray;
    unsigned int i;

    vArray.resize(x.size());

    for( i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    for (int n=0; n<_topology->getNbEdges(); ++n)
    {
        if ((vArray[_topology->getEdge(n)[0]]) && (vArray[_topology->getEdge(n)[1]]))
        {
            // insert a dummy element : computation of pressure done later
            EdgePressureInformation t;
            edgePressureMap[n]=t;
        }
    }
}

template <class DataTypes>
void EdgePressureForceField<DataTypes>::selectEdgesFromString()
{
    std::string inputString=edgeList.getValue();
    unsigned int i;
    do
    {
        const char *str=inputString.c_str();
        for(i=0; (i<inputString.length())&&(str[i]!=','); ++i) ;
        EdgePressureInformation t;

        if (i==inputString.length())
        {
            edgePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i;
        }
        else
        {
            inputString[i]='\0';
            edgePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i+1;
        }
    }
    while (inputString.length()>0);

}
template<class DataTypes>
void EdgePressureForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    const VecCoord& x = *this->mstate->getX();

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    glColor4f(0,1,0,1);

    typename topology::EdgeSubsetData<EdgePressureInformation>::iterator it;

    for(it=edgePressureMap.begin(); it!=edgePressureMap.end(); it++ )
    {
        helper::gl::glVertexT(x[_topology->getEdge((*it).first)[0]]);
        helper::gl::glVertexT(x[_topology->getEdge((*it).first)[1]]);
    }
    glEnd();


    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa
