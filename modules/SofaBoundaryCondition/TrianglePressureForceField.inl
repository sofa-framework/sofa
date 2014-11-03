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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_INL

#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include <SofaBaseTopology/TopologySparseData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
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

template <class DataTypes> TrianglePressureForceField<DataTypes>::~TrianglePressureForceField()
{
}


template <class DataTypes> void TrianglePressureForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();

    _topology = this->getContext()->getMeshTopology();
    if(!_topology)
        serr << "Missing component: Unable to get MeshTopology from the current context. " << sendl;

    if(!_topology)
        return;

    if (dmin.getValue()!=dmax.getValue())
    {
        selectTrianglesAlongPlane();
    }
    if (triangleList.getValue().size()>0)
    {
        selectTrianglesFromString();
    }

    trianglePressureMap.createTopologicalEngine(_topology);
    trianglePressureMap.registerTopologicalData();

    initTriangleInformation();
}

template <class DataTypes>
void TrianglePressureForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& /* d_x */, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    Deriv force;

    //edgePressureMap.activateSubsetData();
    const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();
    const sofa::helper::vector<TrianglePressureInformation>& my_subset = trianglePressureMap.getValue();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        force=my_subset[i].force/3;
        f[_topology->getTriangle(my_map[i])[0]]+=force;
        f[_topology->getTriangle(my_map[i])[1]]+=force;
        f[_topology->getTriangle(my_map[i])[2]]+=force;

    }
    d_f.endEdit();
    updateTriangleInformation();
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */)
{
    //Todo

    //Remove warning
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    (void)kFactor;

    return;
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::initTriangleInformation()
{
    sofa::component::topology::TriangleSetGeometryAlgorithms<DataTypes>* triangleGeo;
    this->getContext()->get(triangleGeo);

    if(!triangleGeo)
        serr << "Missing component: Unable to get TriangleSetGeometryAlgorithms from the current context." << sendl;

    // FIXME: a dirty way to avoid a crash
    if(!triangleGeo)
        return;

    const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();
    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        my_subset[i].area=triangleGeo->computeRestTriangleArea(my_map[i]);
        my_subset[i].force=pressure.getValue()*my_subset[i].area;
    }

    trianglePressureMap.endEdit();
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::updateTriangleInformation()
{
    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();

    for (unsigned int i=0; i<my_subset.size(); ++i)
        my_subset[i].force=(pressure.getValue()*my_subset[i].area);

    trianglePressureMap.endEdit();
}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesAlongPlane()
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    std::vector<bool> vArray;
    unsigned int i;

    vArray.resize(x.size());

    for( i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();
    helper::vector<unsigned int> inputTriangles;

    for (int n=0; n<_topology->getNbTriangles(); ++n)
    {
        if ((vArray[_topology->getTriangle(n)[0]]) && (vArray[_topology->getTriangle(n)[1]])&& (vArray[_topology->getTriangle(n)[2]]) )
        {
            // insert a dummy element : computation of pressure done later
            TrianglePressureInformation t;
            t.area = 0;
            my_subset.push_back(t);
            inputTriangles.push_back(n);
        }
    }
    trianglePressureMap.endEdit();
    trianglePressureMap.setMap2Elements(inputTriangles);

    return;
}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesFromString()
{
    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();
    helper::vector<unsigned int> _triangleList = triangleList.getValue();

    trianglePressureMap.setMap2Elements(_triangleList);

    for (unsigned int i = 0; i < _triangleList.size(); ++i)
    {
        TrianglePressureInformation t;
        t.area = 0;
        my_subset.push_back(t);
    }

    trianglePressureMap.endEdit();

    return;
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!p_showForces.getValue())
        return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    glColor4f(0,1,0,1);

    const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        helper::gl::glVertexT(x[_topology->getTriangle(my_map[i])[0]]);
        helper::gl::glVertexT(x[_topology->getTriangle(my_map[i])[1]]);
        helper::gl::glVertexT(x[_topology->getTriangle(my_map[i])[2]]);
    }
    glEnd();


    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif /* SOFA_NO_OPENGL */
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_INL
