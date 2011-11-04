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
#ifndef SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_INL

#include <sofa/component/forcefield/EdgePressureForceField.h>
#include <sofa/core/visual/VisualParams.h>
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
using namespace core::topology;


template <class DataTypes> EdgePressureForceField<DataTypes>::~EdgePressureForceField()
{
}
// Handle topological changes
template <class DataTypes>
void  EdgePressureForceField<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=_topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=_topology->endChange();


    edgePressureMap.handleTopologyEvents(itBegin,itEnd);

}

template <class DataTypes>
void EdgePressureForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();

    _topology = this->getContext()->getMeshTopology();
    _completeTopology = NULL;
    this->getContext()->get(_completeTopology, core::objectmodel::BaseContext::SearchUp);

    this->getContext()->get(edgeGeo);

    assert(edgeGeo!=0);

    if (edgeGeo==NULL)
    {
        serr << "ERROR(EdgePressureForceField): object must have an EdgeSetTopology."<<sendl;
        return;
    }

    if(_completeTopology == NULL)
    {
        serr << "ERROR(EdgePressureForceField): assume that pressure vector is provdided otherwise TriangleSetTopology is required" << sendl;
    }

    // init edgesubsetData engine
    edgePressureMap.createTopologicalEngine(_completeTopology);
    edgePressureMap.setCreateParameter( (void *) this );
    edgePressureMap.registerTopologicalData();


    if (dmin.getValue()!=dmax.getValue())
    {
        selectEdgesAlongPlane();
    }
    if (edgeList.getValue().size()>0)
    {
        selectEdgesFromString();
    }

    initEdgeInformation();

}


template <class DataTypes>
void EdgePressureForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv &  dataF, const DataVecCoord &  /*dataX */, const DataVecDeriv & /*dataV*/ )
{
    VecDeriv& f = *(dataF.beginEdit());
    Deriv force;

    //edgePressureMap.activateSubsetData();
    const sofa::helper::vector <unsigned int>& my_map = edgePressureMap.getMap2Elements();
    const sofa::helper::vector<EdgePressureInformation>& my_subset = edgePressureMap.getValue();
    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        force=my_subset[i].force/2;
        f[_topology->getEdge(my_map[i])[0]]+=force;
        f[_topology->getEdge(my_map[i])[1]]+=force;
    }

    dataF.endEdit();
    updateEdgeInformation();
}

template<class DataTypes>
void EdgePressureForceField<DataTypes>::initEdgeInformation()
{
    const VecCoord& x = *this->mstate->getX();
    const helper::vector<Real>& intensities = p_intensity.getValue();

    const sofa::helper::vector <unsigned int>& my_map = edgePressureMap.getMap2Elements();
    sofa::helper::vector<EdgePressureInformation>& my_subset = *(edgePressureMap).beginEdit();


    if(pressure.getValue().norm() > 0 )
    {
        for (unsigned int i=0; i<my_map.size(); ++i)
        {
            my_subset[i].length=edgeGeo->computeRestEdgeLength(my_map[i]);
            my_subset[i].force=pressure.getValue()*my_subset[i].length;
        }
    }
    else if (_topology && intensities.size() > 0)
    {
        // binormal provided
        if(p_binormal.isSet())
        {
            Coord binormal = p_binormal.getValue();
            binormal.normalize();
            for(int i = 0; i < _topology->getNbEdges() ; i++)
            {
                Edge e = _topology->getEdge(i);

                Coord tang = x[e[1]] - x[e[0]]; tang.normalize();
                Coord normal = binormal.cross(tang);
                normal.normalize();

                EdgePressureInformation ei;
                Real intensity = (intensities.size() > 1 && intensities.size() > (unsigned int) i) ? intensities[i] : intensities[0];
                //std::cout << intensities.size() << " " << intensities[i] << " " << intensity << std::endl;
                ei.length = edgeGeo->computeRestEdgeLength(i);
                ei.force = normal * intensity * ei.length ;
                edgePressureMap[i] = ei;
            }
        }
        else
            // if no pressure is provided, assume that boundary edges received pressure along their normal
        {
            for(int i = 0; i < _topology->getNbEdges() ; i++)
            {
                Edge e = _topology->getEdge(i), f;

                Vec3d tang, n1, n2;
                n2 = Vec3d(0,0,1);
                tang = x[e[1]] - x[e[0]]; tang.normalize();

                Vec3d sum;
                bool found = false;
                int k = 0;
                while ((!found) && (k < _completeTopology->getNbEdges()))
                {
                    f = _completeTopology->getEdge(k);

                    Vec3d l1 = x[f[0]] - x[e[0]];
                    Vec3d l2 = x[f[1]] - x[e[1]];

                    if((l1.norm() < 1e-6) && (l2.norm() < 1e-6))
                    {
                        found = true;
                    }
                    else
                        k++;

                }

                TrianglesAroundEdge t_a_E = _completeTopology->getTrianglesAroundEdge(k);
                std::cout << "Triangle Around Edge : " << t_a_E.size() << std::endl;

                if(t_a_E.size() == 1) // 2D cases
                {
                    Triangle t = _completeTopology->getTriangle(t_a_E[0]);
                    Vec3d vert;


                    if((t[0] == e[0]) || (t[0] == e[1]))
                    {
                        if((t[1] == e[0]) || (t[1] == e[1]))
                            vert = x[t[2]];
                        else
                            vert = x[t[1]];
                    }
                    else
                        vert = x[t[0]];

                    Vec3d tt = vert - x[e[0]];
                    n1 = n2.cross(tang);
                    if(n1*tt < 0)
                    {
                        n1 = -n1;
                    }

                    EdgePressureInformation ei;
                    Real intensity = (intensities.size() > 1 && intensities.size() > (unsigned int) i) ? intensities[i] : intensities[0];
                    ei.length = edgeGeo->computeRestEdgeLength(i);
                    ei.force = n1 * ei.length * intensity;
                    edgePressureMap[i] = ei;
                }
            }
        }
    }

    edgePressureMap.endEdit();

    return;
}


template<class DataTypes>
void EdgePressureForceField<DataTypes>::updateEdgeInformation()
{
    const VecCoord& x = *this->mstate->getX();

    const sofa::helper::vector <unsigned int>& my_map = edgePressureMap.getMap2Elements();
    sofa::helper::vector<EdgePressureInformation>& my_subset = *(edgePressureMap).beginEdit();
    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        Vec3d p1 = x[_topology->getEdge(my_map[i])[0]];
        Vec3d p2 = x[_topology->getEdge(my_map[i])[1]];
        Vec3d orig(0,0,0);

        Vec3d tang = p2 - p1;
        tang.norm();

        Deriv myPressure;

        if( (p1[0] - orig[0]) * tang[1] > 0)
            myPressure[0] = tang[1];
        else
            myPressure[0] = - tang[1];

        if( (p1[1] - orig[1]) * tang[0] > 0)
            myPressure[1] = tang[0];
        else
            myPressure[1] = - tang[0];

        //my_subset[i].force=pressure.getValue()*(my_subset[i].length);
        my_subset[i].force=myPressure*(my_subset[i].length);

    }
    edgePressureMap.endEdit();
    initEdgeInformation();
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

    sofa::helper::vector<EdgePressureInformation>& my_subset = *(edgePressureMap).beginEdit();
    helper::vector<unsigned int> inputEdges;


    for (int n=0; n<_topology->getNbEdges(); ++n)
    {
        if ((vArray[_topology->getEdge(n)[0]]) && (vArray[_topology->getEdge(n)[1]]))
        {
            // insert a dummy element : computation of pressure done later
            EdgePressureInformation t;
            my_subset.push_back(t);
            inputEdges.push_back(n);
        }
    }
    edgePressureMap.endEdit();
    edgePressureMap.setMap2Elements(inputEdges);

    return;
}

template <class DataTypes>
void EdgePressureForceField<DataTypes>::selectEdgesFromString()
{
    const helper::vector<unsigned int>& inputString = edgeList.getValue();
    edgePressureMap.setMap2Elements(inputString);

    sofa::helper::vector<EdgePressureInformation>& my_subset = *(edgePressureMap).beginEdit();

    for (unsigned int i = 0; i < inputString.size(); ++i)
    {
        EdgePressureInformation t;
        my_subset.push_back(t);
    }
    edgePressureMap.endEdit();

    return;
}

template<class DataTypes>
void EdgePressureForceField<DataTypes>::draw(const core::visual::VisualParams*)
{
    if (!p_showForces.getValue())
        return;

    double aSC = arrowSizeCoef.getValue();

    const VecCoord& x = *this->mstate->getX();
    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    glColor4f(1,1,0,1);

    const sofa::helper::vector <unsigned int>& my_map = edgePressureMap.getMap2Elements();
    const sofa::helper::vector<EdgePressureInformation>& my_subset = edgePressureMap.getValue();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        Vec3d p = (x[_topology->getEdge(my_map[i])[0]] + x[_topology->getEdge(my_map[i])[1]]) / 2.0;
        helper::gl::glVertexT(p);

        Vec3d f = my_subset[i].force;
        //f.normalize();
        f *= aSC;
        helper::gl::glVertexT(p + f);
    }
    glEnd();

}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_INL
