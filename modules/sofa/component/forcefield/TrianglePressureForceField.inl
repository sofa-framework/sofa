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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_INL

#include <sofa/component/forcefield/TrianglePressureForceField.h>
#include <sofa/component/topology/TriangleSubsetData.inl>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <set>

// #define DEBUG_TRIANGLEFEM

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::topology;






template <class DataTypes> TrianglePressureForceField<DataTypes>::~TrianglePressureForceField()
{
}
// Handle topological changes
template <class DataTypes> void  TrianglePressureForceField<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=_topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=_topology->endChange();


    trianglePressureMap.handleTopologyEvents(itBegin,itEnd,_topology->getNbTriangles());

}
template <class DataTypes> void TrianglePressureForceField<DataTypes>::init()
{
    //serr << "initializing TrianglePressureForceField" << sendl;
    this->core::behavior::ForceField<DataTypes>::init();

    _topology = this->getContext()->getMeshTopology();

    if (dmin.getValue()!=dmax.getValue())
    {
        selectTrianglesAlongPlane();
    }
    if (triangleList.getValue().length()>0)
    {
        selectTrianglesFromString();
    }

    initTriangleInformation();

}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::addForce(DataVecDeriv& d_f, const DataVecCoord& /* d_x */, const DataVecDeriv& /* d_v */, const core::MechanicalParams* /* mparams */)
{
    VecDeriv& f = *d_f.beginEdit();
    Deriv force;

    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        force=(*it).second.force/3;
        f[_topology->getTriangle((*it).first)[0]]+=force;
        f[_topology->getTriangle((*it).first)[1]]+=force;
        f[_topology->getTriangle((*it).first)[2]]+=force;

    }
    d_f.endEdit();
}

template<class DataTypes>
void TrianglePressureForceField<DataTypes>::addDForce(DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */, const core::MechanicalParams* /* mparams */)
{
    //Todo
}

template<class DataTypes>
void TrianglePressureForceField<DataTypes>::initTriangleInformation()
{
    sofa::component::topology::TriangleSetGeometryAlgorithms<DataTypes>* triangleGeo;
    this->getContext()->get(triangleGeo);

    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        (*it).second.area=triangleGeo->computeRestTriangleArea((*it).first);
        (*it).second.force=pressure.getValue()*(*it).second.area;
    }
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::updateTriangleInformation()
{
    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        (*it).second.force=(pressure.getValue()*(*it).second.area);
    }
}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesAlongPlane()
{
    const VecCoord& x = *this->mstate->getX0();
    std::vector<bool> vArray;
    unsigned int i;

    vArray.resize(x.size());

    for( i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    for (int n=0; n<_topology->getNbTriangles(); ++n)
    {
        if ((vArray[_topology->getTriangle(n)[0]]) && (vArray[_topology->getTriangle(n)[1]])&& (vArray[_topology->getTriangle(n)[2]]) )
        {
            // insert a dummy element : computation of pressure done later
            TrianglePressureInformation t;
            t.area = 0;
            trianglePressureMap[n]=t;
        }
    }
}

template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesFromString()
{
    std::string inputString=triangleList.getValue();
    unsigned int i;
    do
    {
        const char *str=inputString.c_str();
        for(i=0; (i<inputString.length())&&(str[i]!=','); ++i) ;
        TrianglePressureInformation t;
        t.area = 0;
        if (i==inputString.length())
        {
            trianglePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i;
        }
        else
        {
            inputString[i]='\0';
            trianglePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i+1;
        }
    }
    while (inputString.length()>0);

}
template<class DataTypes>
void TrianglePressureForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    const VecCoord& x = *this->mstate->getX();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    glColor4f(0,1,0,1);

    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        helper::gl::glVertexT(x[_topology->getTriangle((*it).first)[0]]);
        helper::gl::glVertexT(x[_topology->getTriangle((*it).first)[1]]);
        helper::gl::glVertexT(x[_topology->getTriangle((*it).first)[2]]);
    }
    glEnd();


    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_INL
