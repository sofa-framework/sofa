/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_INL

#include <SofaBoundaryCondition/EdgePressureForceField.h>
#include <SofaBaseTopology/TopologySparseData.inl>
#include <sofa/core/visual/VisualParams.h>
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


template <class DataTypes> EdgePressureForceField<DataTypes>::~EdgePressureForceField()
{
}


template <class DataTypes>
void EdgePressureForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();

    _topology = this->getContext()->getMeshTopology();
    if(_topology == NULL)
    {
        serr << "ERROR(EdgePressureForceField): No base topology available." << sendl;
        return;
    }

    this->getContext()->get(edgeGeo);
    assert(edgeGeo!=0);

    if (edgeGeo==NULL)
    {
        serr << "ERROR(EdgePressureForceField): object must have an EdgeSetTopology."<<sendl;
        return;
    }


    _completeTopology = NULL;
    this->getContext()->get(_completeTopology, core::objectmodel::BaseContext::SearchUp);

    if(_completeTopology == NULL && edgeIndices.getValue().empty() && edges.getValue().empty())
    {
        serr << "ERROR(EdgePressureForceField): Either a pressure vector or a TriangleSetTopology is required." << sendl;
    }

    // init edgesubsetData engine
    edgePressureMap.createTopologicalEngine(_topology);
    edgePressureMap.registerTopologicalData();

    if (dmin.getValue()!=dmax.getValue())
    {
        selectEdgesAlongPlane();
    }
    if (edgeIndices.getValue().size()>0)
    {
        selectEdgesFromString();
    }
    if (edges.getValue().size()>0)
    {
        selectEdgesFromEdgeList();
    }

    initEdgeInformation();
}


template <class DataTypes>
void EdgePressureForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  /*dataX */, const DataVecDeriv & /*dataV*/ )
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
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (x.empty())
    {
        serr << "ERROR(EdgePressureForceField): No mechanical Object linked."<<sendl;
        return;
    }

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
            for(unsigned int i = 0; i < my_map.size() ; i++)
            {
                core::topology::BaseMeshTopology::Edge e = _topology->getEdge(my_map[i]);  // FF,13/03/2012: This seems more consistent

                Coord tang = x[e[1]] - x[e[0]]; tang.normalize();
                Coord normal = binormal.cross(tang);
                normal.normalize();

                EdgePressureInformation ei;
                Real intensity = (intensities.size() > 1 && intensities.size() > (unsigned int) i) ? intensities[i] : intensities[0];
                ei.length = edgeGeo->computeRestEdgeLength(i);
                ei.force = normal * intensity * ei.length ;
                edgePressureMap[i] = ei;
            }
        }
        else
            // if no pressure is provided, assume that boundary edges received pressure along their normal
        {
            for(unsigned i = 0; i < my_map.size() ; i++)
            {
                core::topology::BaseMeshTopology::Edge e = _topology->getEdge(my_map[i]), f;

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

                core::topology::BaseMeshTopology::TrianglesAroundEdge t_a_E = _completeTopology->getTrianglesAroundEdge(k);

                if(t_a_E.size() == 1) // 2D cases
                {
                    core::topology::BaseMeshTopology::Triangle t = _completeTopology->getTriangle(t_a_E[0]);
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
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (x.empty())
    {
        serr << "ERROR(EdgePressureForceField): No mechanical Object linked."<<sendl;
        return;
    }

    const sofa::helper::vector <unsigned int>& my_map = edgePressureMap.getMap2Elements();
    sofa::helper::vector<EdgePressureInformation>& my_subset = *(edgePressureMap).beginEdit();
    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        sofa::defaulttype::Vec3d p1 = x[_topology->getEdge(my_map[i])[0]];
        sofa::defaulttype::Vec3d p2 = x[_topology->getEdge(my_map[i])[1]];
        sofa::defaulttype::Vec3d orig(0,0,0);

        sofa::defaulttype::Vec3d tang = p2 - p1;
        tang.norm(); /// @todo: shouldn't this be normalize() ?

        Deriv myPressure;

        if( (p1[0] - orig[0]) * tang[1] > 0)
            myPressure[0] = (Real)tang[1];
        else
            myPressure[0] = - (Real)tang[1];

        if( (p1[1] - orig[1]) * tang[0] > 0)
            myPressure[1] = (Real)tang[0];
        else
            myPressure[1] = - (Real)tang[0];

        //my_subset[i].force=pressure.getValue()*(my_subset[i].length);
        my_subset[i].force=myPressure*(my_subset[i].length);

    }
    edgePressureMap.endEdit();
    initEdgeInformation();
}


template <class DataTypes>
void EdgePressureForceField<DataTypes>::selectEdgesAlongPlane()
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
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
void EdgePressureForceField<DataTypes>::selectEdgesFromIndices(const helper::vector<unsigned int>& inputIndices)
{
    edgePressureMap.setMap2Elements(inputIndices);

    sofa::helper::vector<EdgePressureInformation>& my_subset = *(edgePressureMap).beginEdit();

    unsigned int sizeTest = _topology->getNbEdges();

    for (unsigned int i = 0; i < inputIndices.size(); ++i)
    {
        EdgePressureInformation t;
        my_subset.push_back(t);

        if (inputIndices[i] >= sizeTest)
            serr << "ERROR(EdgePressureForceField): Edge indice: " << inputIndices[i] << " is out of edge indices bounds. This could lead to non desired behavior." <<sendl;
    }
    edgePressureMap.endEdit();

    return;
}

template <class DataTypes>
void EdgePressureForceField<DataTypes>::selectEdgesFromString()
{
    const helper::vector<unsigned int>& inputString = edgeIndices.getValue();
    selectEdgesFromIndices(inputString);
}

template<class DataTypes>
void EdgePressureForceField<DataTypes>::selectEdgesFromEdgeList()
{
    const helper::vector<core::topology::BaseMeshTopology::Edge>& inputEdges = edges.getValue();
    const helper::vector<core::topology::BaseMeshTopology::Edge>& topologyEdges = _topology->getEdges();

    helper::vector<unsigned int> indices(inputEdges.size());

    for(unsigned int i=0; i<inputEdges.size(); i++)
    {
        core::topology::BaseMeshTopology::Edge inputEdge = inputEdges[i];
        for(unsigned int j=0; j<topologyEdges.size(); j++)
        {
            core::topology::BaseMeshTopology::Edge topologyEdge = topologyEdges[j];
            //If they are the same edge
            if(inputEdge[0] == topologyEdge[0] && inputEdge[1] == topologyEdge[1])
            {
                indices[i] = j;
            }
        }
    }

    selectEdgesFromIndices(indices);
}

template<class DataTypes>
void EdgePressureForceField<DataTypes>::draw(const core::visual::VisualParams*)
{
#ifndef SOFA_NO_OPENGL
    if (!p_showForces.getValue())
        return;

    SReal aSC = arrowSizeCoef.getValue();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    glColor4f(1,1,0,1);

    const sofa::helper::vector <unsigned int>& my_map = edgePressureMap.getMap2Elements();
    const sofa::helper::vector<EdgePressureInformation>& my_subset = edgePressureMap.getValue();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        sofa::defaulttype::Vec3d p = (x[_topology->getEdge(my_map[i])[0]] + x[_topology->getEdge(my_map[i])[1]]) / 2.0;
        sofa::helper::gl::glVertexT(p);

        sofa::defaulttype::Vec3d f = my_subset[i].force;
        //f.normalize();
        f *= aSC;
        helper::gl::glVertexT(p + f);
    }
    glEnd();
#endif /* SOFA_NO_OPENGL */
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_INL
