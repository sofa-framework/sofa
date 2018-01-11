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
#ifndef SOFA_COMPONENT_ENGINE_ROTATABLEBOXROI_INL
#define SOFA_COMPONENT_ENGINE_ROTATABLEBOXROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/RotatableBoxROI.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <limits>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/simulation/AnimateBeginEvent.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
RotatableBoxROI<DataTypes>::RotatableBoxROI()
    : d_alignedBoxes( initData(&d_alignedBoxes, "box", "List of boxes defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , d_rotations( initData(&d_rotations, "rotation", "List of rotation defined in Euler angles in degree") )
    , d_X0( initData (&d_X0, "position", "Rest position coordinates of the degrees of freedom. \n"
                                         "If empty the positions from a MechanicalObject then a MeshLoader are searched in the current context. \n"
                                         "If none are found the parent's context is searched for MechanicalObject." ) )
    , d_edges(initData (&d_edges, "edges", "Edge Topology") )
    , d_triangles(initData (&d_triangles, "triangles", "Triangle Topology") )
    , d_tetrahedra(initData (&d_tetrahedra, "tetrahedra", "Tetrahedron Topology") )
    , d_hexahedra(initData (&d_hexahedra, "hexahedra", "Hexahedron Topology") )
    , d_quad(initData (&d_quad, "quad", "Quad Topology") )
    , d_computeEdges( initData(&d_computeEdges, true,"computeEdges","If true, will compute edge list and index list inside the ROI. (default = true)") )
    , d_computeTriangles( initData(&d_computeTriangles, true,"computeTriangles","If true, will compute triangle list and index list inside the ROI. (default = true)") )
    , d_computeTetrahedra( initData(&d_computeTetrahedra, true,"computeTetrahedra","If true, will compute tetrahedra list and index list inside the ROI. (default = true)") )
    , d_computeHexahedra( initData(&d_computeHexahedra, true,"computeHexahedra","If true, will compute hexahedra list and index list inside the ROI. (default = true)") )
    , d_computeQuad( initData(&d_computeQuad, true,"computeQuad","If true, will compute quad list and index list inside the ROI. (default = true)") )
    , d_indices( initData(&d_indices,"indices","Indices of the points contained in the ROI") )
    , d_edgeIndices( initData(&d_edgeIndices,"edgeIndices","Indices of the edges contained in the ROI") )
    , d_triangleIndices( initData(&d_triangleIndices,"triangleIndices","Indices of the triangles contained in the ROI") )
    , d_tetrahedronIndices( initData(&d_tetrahedronIndices,"tetrahedronIndices","Indices of the tetrahedra contained in the ROI") )
    , d_hexahedronIndices( initData(&d_hexahedronIndices,"hexahedronIndices","Indices of the hexahedra contained in the ROI") )
    , d_quadIndices( initData(&d_quadIndices,"quadIndices","Indices of the quad contained in the ROI") )
    , d_pointsInROI( initData(&d_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , d_edgesInROI( initData(&d_edgesInROI,"edgesInROI","Edges contained in the ROI") )
    , d_trianglesInROI( initData(&d_trianglesInROI,"trianglesInROI","Triangles contained in the ROI") )
    , d_tetrahedraInROI( initData(&d_tetrahedraInROI,"tetrahedraInROI","Tetrahedra contained in the ROI") )
    , d_hexahedraInROI( initData(&d_hexahedraInROI,"hexahedraInROI","Hexahedra contained in the ROI") )
    , d_quadInROI( initData(&d_quadInROI,"quadInROI","Quad contained in the ROI") )
    , d_nbIndices( initData(&d_nbIndices,"nbIndices", "Number of selected indices") )
    , d_drawBoxes( initData(&d_drawBoxes,false,"drawBoxes","Draw Boxes. (default = false)") )
    , d_drawPoints( initData(&d_drawPoints,false,"drawPoints","Draw Points. (default = false)") )
    , d_drawEdges( initData(&d_drawEdges,false,"drawEdges","Draw Edges. (default = false)") )
    , d_drawTriangles( initData(&d_drawTriangles,false,"drawTriangles","Draw Triangles. (default = false)") )
    , d_drawTetrahedra( initData(&d_drawTetrahedra,false,"drawTetrahedra","Draw Tetrahedra. (default = false)") )
    , d_drawHexahedra( initData(&d_drawHexahedra,false,"drawHexahedra","Draw Tetrahedra. (default = false)") )
    , d_drawQuads( initData(&d_drawQuads,false,"drawQuads","Draw Quads. (default = false)") )
    , d_drawSize( initData(&d_drawSize,0.0,"drawSize","rendering size for box and topological elements") )
    , d_doUpdate( initData(&d_doUpdate,(bool)true,"doUpdate","If true, updates the selection at the beginning of simulation steps. (default = true)") )

    /// In case you add a new attribute please also add it into to the BoxROI_test.cpp::attributesTests
    /// In case you want to remove or rename an attribute, please keep it as-is but add a warning message
    /// using msg_warning saying to the user of this component that the attribute is deprecated and solutions/replacement
    /// he has to fix his scene.
{
    //Adding alias to handle old BoxROI outputs
    addAlias(&d_pointsInROI,"pointsInBox");
    addAlias(&d_edgesInROI,"edgesInBox");
    addAlias(&d_trianglesInROI,"f_trianglesInBox");
    addAlias(&d_tetrahedraInROI,"f_tetrahedraInBox");
    addAlias(&d_hexahedraInROI,"f_tetrahedraInBox");
    addAlias(&d_quadInROI,"f_quadInBOX");

    if(!d_alignedBoxes.isSet())
    {
        d_alignedBoxes.beginEdit()->push_back(Vec6(0,0,0,1,1,1));
        d_alignedBoxes.endEdit();
        d_rotations.beginEdit()->push_back(Vec3(0,0,0));
        d_rotations.endEdit();
    }

    d_indices.beginEdit()->push_back(0);
    d_indices.endEdit();
}

template <class DataTypes>
void RotatableBoxROI<DataTypes>::init()
{
    using sofa::core::objectmodel::BaseData;
    using sofa::core::topology::BaseMeshTopology;
    using sofa::core::objectmodel::BaseContext;

    /// If the position attribute is not set we are trying to
    /// automatically load the positions from the current context MechanicalState if any, then
    /// in a MeshLoad if any and in case of failure it will finally search it in the parent's
    /// context.
    if (!d_X0.isSet())
    {
        msg_info(this) << "No attribute 'position' set.\n"
                          "Searching in the context for a MechanicalObject or MeshLoader.\n" ;

        sofa::core::behavior::MechanicalState<DataTypes>* mstate = nullptr ;
        this->getContext()->get(mstate, BaseContext::Local);
        if (mstate)
        {
            BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                d_X0.setParent(parent);
                d_X0.setReadOnly(true);
            }else{
                msg_warning(this) << "No attribute 'rest_position' in component '" << getName() << "'.\n"
                                  << "The RotatableBoxROI component thus have no input and is thus deactivated.\n" ;
                m_componentstate = core::objectmodel::ComponentState::Invalid;
                return ;
            }
        }
        else
        {
            core::loader::MeshLoader* loader = nullptr ;
            this->getContext()->get(loader, BaseContext::Local);
            if (loader)
            {
                BaseData* parent = loader->findData("position");
                if (parent)
                {
                    d_X0.setParent(parent);
                    d_X0.setReadOnly(true);
                }else{
                    msg_warning(this) << "No attribute 'position' in component '" << getName() << "'.\n"
                                      << "The BoxROI component thus have no input and is thus deactivated.\n" ;
                    m_componentstate = core::objectmodel::ComponentState::Invalid ;
                    return ;
                }
            }
            else   // no local state, no loader => find upward
            {
                this->getContext()->get(mstate, BaseContext::SearchUp);
                if(!mstate){
                    msg_error(this) <<  "Unable to find a MechanicalObject for this component. "
                                        "To remove this error message you can either:\n"
                                        "   - to specifiy the DOF where to apply the BoxROI with the 'position' attribute.\n"
                                        "   - to add MechanicalObject or MeshLoader component before the BoxROI in the scene graph.\n";
                    m_componentstate = core::objectmodel::ComponentState::Invalid ;
                    return ;
                }

                BaseData* parent = mstate->findData("rest_position");
                if(!parent){
                    dmsg_error(this) <<  "Unable to find a rest_position attribute in the MechanicalObject '" << mstate->getName() << "'";
                    m_componentstate = core::objectmodel::ComponentState::Invalid ;
                    return ;
                }
                d_X0.setParent(parent);
                d_X0.setReadOnly(true);
            }
        }
    }


    if (!d_edges.isSet() || !d_triangles.isSet() || !d_tetrahedra.isSet() || !d_hexahedra.isSet() || !d_quad.isSet() )
    {
        msg_info(this) << "No topology given. Searching for a TopologyContainer and a BaseMeshTopology in the current context.\n";

        core::topology::TopologyContainer* topologyContainer;
        this->getContext()->get(topologyContainer, BaseContext::Local);

        BaseMeshTopology* topology;
        this->getContext()->get(topology, BaseContext::Local);

        if (topologyContainer || topology)
        {
            if (!d_edges.isSet() && d_computeEdges.getValue())
            {
                BaseData* eparent = topologyContainer?topologyContainer->findData("edges"):topology->findData("edges");
                if (eparent)
                {
                    d_edges.setParent(eparent);
                    d_edges.setReadOnly(true);
                }
            }
            if (!d_triangles.isSet() && d_computeTriangles.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("triangles"):topology->findData("triangles");
                if (tparent)
                {
                    d_triangles.setParent(tparent);
                    d_triangles.setReadOnly(true);
                }
            }
            if (!d_tetrahedra.isSet() && d_computeTetrahedra.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("tetrahedra"):topology->findData("tetrahedra");
                if (tparent)
                {
                    d_tetrahedra.setParent(tparent);
                    d_tetrahedra.setReadOnly(true);
                }
            }
            if (!d_hexahedra.isSet() && d_computeHexahedra.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("hexahedra"):topology->findData("hexahedra");
                if (tparent)
                {
                    d_hexahedra.setParent(tparent);
                    d_hexahedra.setReadOnly(true);
                }
            }
            if (!d_quad.isSet() && d_computeQuad.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("quads"):topology->findData("quads");
                if (tparent)
                {
                    d_quad.setParent(tparent);
                    d_quad.setReadOnly(true);
                }
            }
        }
    }

    addInput(&d_X0);
    addInput(&d_edges);
    addInput(&d_triangles);
    addInput(&d_tetrahedra);
    addInput(&d_hexahedra);
    addInput(&d_quad);

    addOutput(&d_indices);
    addOutput(&d_edgeIndices);
    addOutput(&d_triangleIndices);
    addOutput(&d_tetrahedronIndices);
    addOutput(&d_hexahedronIndices);
    addOutput(&d_quadIndices);
    addOutput(&d_pointsInROI);
    addOutput(&d_edgesInROI);
    addOutput(&d_trianglesInROI);
    addOutput(&d_tetrahedraInROI);
    addOutput(&d_hexahedraInROI);
    addOutput(&d_quadInROI);
    addOutput(&d_nbIndices);

    m_componentstate = core::objectmodel::ComponentState::Valid ;

    /// The following is a trick to force the initial selection of the element by the engine.
    bool tmp=d_doUpdate.getValue() ;
    d_doUpdate.setValue(true);
    setDirtyValue();
    initialiseBoxes();
    d_doUpdate.setValue(tmp);
}


template <class DataTypes>
void RotatableBoxROI<DataTypes>::initialiseBoxes()
{
    helper::vector<Vec6>& alignedBoxes = *(d_alignedBoxes.beginEdit());
    if (!alignedBoxes.empty())
    {
        for (unsigned int bi=0; bi<alignedBoxes.size(); ++bi)
        {
            if (alignedBoxes[bi][0] > alignedBoxes[bi][3]) std::swap(alignedBoxes[bi][0],alignedBoxes[bi][3]);
            if (alignedBoxes[bi][1] > alignedBoxes[bi][4]) std::swap(alignedBoxes[bi][1],alignedBoxes[bi][4]);
            if (alignedBoxes[bi][2] > alignedBoxes[bi][5]) std::swap(alignedBoxes[bi][2],alignedBoxes[bi][5]);
        }
    }
    d_alignedBoxes.endEdit();

    computeRotatedBoxes();

    update();
}


template <class DataTypes>
void RotatableBoxROI<DataTypes>::computeRotatedBoxes()
{
    // take the aligned boxes and aplly rotation to calculate the rotated boxes
    helper::ReadAccessor<Data<helper::vector<Vec6> > > alignedBoxes = d_alignedBoxes;
    helper::ReadAccessor<Data<helper::vector<Vec3> > > rotations = d_rotations;

    if(alignedBoxes.empty())
        return;

    m_rotatedBoxes.resize(alignedBoxes.size());

    Vec4 tempPlane;
    for(unsigned int index = 0; index < alignedBoxes.size(); index++)
    {
        // calculate quaternion
        sofa::defaulttype::Quaternion rotationQuat =
                helper::Quater<Real>::createQuaterFromEuler(rotations[index] * M_PI / 180.0);

        // calculate center
        Vec3 alignedBoxCenter;
        alignedBoxCenter[0] = 0.5 * (alignedBoxes[index][0] + alignedBoxes[index][3]);
        alignedBoxCenter[1] = 0.5 * (alignedBoxes[index][1] + alignedBoxes[index][4]);
        alignedBoxCenter[2] = 0.5 * (alignedBoxes[index][2] + alignedBoxes[index][5]);

        // create upper plane
        Vec3 centerPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][4], alignedBoxes[index][5]);
        Vec3 firstVecPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][4], alignedBoxes[index][2]);
        Vec3 secondVecPlanePoint = Vec3(alignedBoxes[index][3], alignedBoxes[index][4], alignedBoxes[index][5]);
        rotatePlane(centerPlanePoint, firstVecPlanePoint, secondVecPlanePoint, alignedBoxCenter, rotationQuat, tempPlane);
        m_rotatedBoxes[index].upperPlane = tempPlane;

        // create lower plane
        centerPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][1], alignedBoxes[index][5]);
        firstVecPlanePoint = Vec3(alignedBoxes[index][3], alignedBoxes[index][1], alignedBoxes[index][5]);
        secondVecPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][1], alignedBoxes[index][2]);
        rotatePlane(centerPlanePoint, firstVecPlanePoint, secondVecPlanePoint, alignedBoxCenter, rotationQuat, tempPlane);
        m_rotatedBoxes[index].lowerPlane = tempPlane;

        // create front plane
        centerPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][1], alignedBoxes[index][2]);
        firstVecPlanePoint = Vec3(alignedBoxes[index][3], alignedBoxes[index][1], alignedBoxes[index][2]);
        secondVecPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][4], alignedBoxes[index][2]);
        rotatePlane(centerPlanePoint, firstVecPlanePoint, secondVecPlanePoint, alignedBoxCenter, rotationQuat, tempPlane);
        m_rotatedBoxes[index].frontPlane = tempPlane;

        // create back plane
        centerPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][1], alignedBoxes[index][5]);
        firstVecPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][4], alignedBoxes[index][5]);
        secondVecPlanePoint = Vec3(alignedBoxes[index][3], alignedBoxes[index][1], alignedBoxes[index][5]);
        rotatePlane(centerPlanePoint, firstVecPlanePoint, secondVecPlanePoint, alignedBoxCenter, rotationQuat, tempPlane);
        m_rotatedBoxes[index].backPlane = tempPlane;

        // create left plane
        centerPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][1], alignedBoxes[index][5]);
        firstVecPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][1], alignedBoxes[index][2]);
        secondVecPlanePoint = Vec3(alignedBoxes[index][0], alignedBoxes[index][4], alignedBoxes[index][5]);
        rotatePlane(centerPlanePoint, firstVecPlanePoint, secondVecPlanePoint, alignedBoxCenter, rotationQuat, tempPlane);
        m_rotatedBoxes[index].leftPlane = tempPlane;

        // create right plane
        centerPlanePoint = Vec3(alignedBoxes[index][3], alignedBoxes[index][1], alignedBoxes[index][5]);
        firstVecPlanePoint = Vec3(alignedBoxes[index][3], alignedBoxes[index][4], alignedBoxes[index][5]);
        secondVecPlanePoint = Vec3(alignedBoxes[index][3], alignedBoxes[index][1], alignedBoxes[index][2]);
        rotatePlane(centerPlanePoint, firstVecPlanePoint, secondVecPlanePoint, alignedBoxCenter, rotationQuat, tempPlane);
        m_rotatedBoxes[index].rightPlane = tempPlane;

        rotateBoxPoints(alignedBoxes[index], alignedBoxCenter, rotationQuat, &(m_rotatedBoxes[index].boxPoints));
    }
}


template <class DataTypes>
void RotatableBoxROI<DataTypes>::rotatePlane(Vec3& centerPlanePoint, Vec3& firstPlanePoint, Vec3& secondPlanePoint,
                                             Vec3& centerShift, const defaulttype::Quaternion& rotationData, Vec4& resPlane)
{
    Vec3 temporaryPoint;

    // shift plane points to the center
    centerPlanePoint -= centerShift;
    firstPlanePoint -= centerShift;
    secondPlanePoint -= centerShift;

    // rotate points
    temporaryPoint = rotationData.rotate(centerPlanePoint);
    centerPlanePoint = temporaryPoint;
    temporaryPoint = rotationData.rotate(firstPlanePoint);
    firstPlanePoint = temporaryPoint;
    temporaryPoint = rotationData.rotate(secondPlanePoint);
    secondPlanePoint = temporaryPoint;

    // shift plane point back
    centerPlanePoint += centerShift;
    firstPlanePoint += centerShift;
    secondPlanePoint += centerShift;

    // generate plane
    Vec3 planeNormal = sofa::defaulttype::cross(firstPlanePoint - centerPlanePoint, secondPlanePoint - centerPlanePoint).normalized();
    Real planeShift = -sofa::defaulttype::dot(planeNormal, firstPlanePoint);
    resPlane = Vec4(planeNormal[0], planeNormal[1], planeNormal[2], planeShift);
}


template <class DataTypes>
void RotatableBoxROI<DataTypes>::rotateBoxPoints(const Vec6& alignedBox, Vec3& centerShift, defaulttype::Quaternion& rotationData, helper::vector<Vec3> *resPoints)
{
    // add eight box points
    Vec3 tempPoint;
    rotatePoint(Vec3(alignedBox[0], alignedBox[1], alignedBox[2]), centerShift, rotationData, tempPoint);
    resPoints->push_back(tempPoint);
    rotatePoint(Vec3(alignedBox[0], alignedBox[1], alignedBox[5]), centerShift, rotationData, tempPoint);
    resPoints->push_back(tempPoint);
    rotatePoint(Vec3(alignedBox[3], alignedBox[1], alignedBox[5]), centerShift, rotationData, tempPoint);
    resPoints->push_back(tempPoint);
    rotatePoint(Vec3(alignedBox[3], alignedBox[1], alignedBox[2]), centerShift, rotationData, tempPoint);
    resPoints->push_back(tempPoint);
    rotatePoint(Vec3(alignedBox[0], alignedBox[4], alignedBox[2]), centerShift, rotationData, tempPoint);
    resPoints->push_back(tempPoint);
    rotatePoint(Vec3(alignedBox[0], alignedBox[4], alignedBox[5]), centerShift, rotationData, tempPoint);
    resPoints->push_back(tempPoint);
    rotatePoint(Vec3(alignedBox[3], alignedBox[4], alignedBox[5]), centerShift, rotationData, tempPoint);
    resPoints->push_back(tempPoint);
    rotatePoint(Vec3(alignedBox[3], alignedBox[4], alignedBox[2]), centerShift, rotationData, tempPoint);
    resPoints->push_back(tempPoint);
}


template <class DataTypes>
void RotatableBoxROI<DataTypes>::rotatePoint(Vec3 initialPoint, Vec3& centerShift, const defaulttype::Quaternion& rotationData, Vec3& resPoint)
{
    Vec3 temporaryPoint;
    initialPoint -= centerShift;
    temporaryPoint = rotationData.rotate(initialPoint);
    initialPoint = temporaryPoint;
    initialPoint += centerShift;
    resPoint = initialPoint;
}


template <class DataTypes>
void RotatableBoxROI<DataTypes>::reinit()
{
    update();
}


template <class DataTypes>
bool RotatableBoxROI<DataTypes>::inNegativeHalfOfSpace(const typename DataTypes::CPos& point, const Vec4& plane)
{
    return sofa::defaulttype::dot(Vec3(point), Vec3(plane[0], plane[1], plane[2])) + plane[3] >= 0;
}


template <class DataTypes>
bool RotatableBoxROI<DataTypes>::isPointInRotatedBox(const typename DataTypes::CPos& point, const RotatedBox &box)
{
    if(inNegativeHalfOfSpace(point, box.upperPlane) && inNegativeHalfOfSpace(point, box.lowerPlane) &&
       inNegativeHalfOfSpace(point, box.frontPlane) && inNegativeHalfOfSpace(point, box.backPlane) &&
            inNegativeHalfOfSpace(point, box.leftPlane) && inNegativeHalfOfSpace(point, box.rightPlane)) {
        return true;
    } else {
        return false;
    }
}


template <class DataTypes>
bool RotatableBoxROI<DataTypes>::isPointInBoxes(const typename DataTypes::CPos& p)
{
    const helper::vector<RotatedBox>& rotatedBoxes = m_rotatedBoxes;

    for (unsigned int i = 0; i < rotatedBoxes.size(); ++i)
        if (isPointInRotatedBox(p, rotatedBoxes[i]))
            return true;

    return false;
}


template <class DataTypes>
bool RotatableBoxROI<DataTypes>::isPointInBoxes(const PointID& pid)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p =  DataTypes::getCPos(x0[pid]);
    return ( isPointInBoxes(p) );
}


template <class DataTypes>
bool RotatableBoxROI<DataTypes>::isEdgeInBoxes(const Edge& e)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[e[0]]);
    CPos p1 =  DataTypes::getCPos(x0[e[1]]);
    CPos c = (p1+p0)*0.5;

    return isPointInBoxes(c);
}


template <class DataTypes>
bool RotatableBoxROI<DataTypes>::isTriangleInBoxes(const Triangle& t)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos c = (p2+p1+p0)/3.0;

    return (isPointInBoxes(c));
}


template <class DataTypes>
bool RotatableBoxROI<DataTypes>::isTetrahedronInBoxes(const Tetra &t)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos p3 =  DataTypes::getCPos(x0[t[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (isPointInBoxes(c));
}


template <class DataTypes>
bool RotatableBoxROI<DataTypes>::isHexahedronInBoxes(const Hexa &t)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos p3 =  DataTypes::getCPos(x0[t[3]]);
    CPos p4 =  DataTypes::getCPos(x0[t[4]]);
    CPos p5 =  DataTypes::getCPos(x0[t[5]]);
    CPos p6 =  DataTypes::getCPos(x0[t[6]]);
    CPos p7 =  DataTypes::getCPos(x0[t[7]]);
    CPos c = (p7+p6+p5+p4+p3+p2+p1+p0)/8.0;

    return (isPointInBoxes(c));
}


template <class DataTypes>
bool RotatableBoxROI<DataTypes>::isQuadInBoxes(const Quad& q)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[q[0]]);
    CPos p1 =  DataTypes::getCPos(x0[q[1]]);
    CPos p2 =  DataTypes::getCPos(x0[q[2]]);
    CPos p3 =  DataTypes::getCPos(x0[q[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (isPointInBoxes(c));

}


template <class DataTypes>
void RotatableBoxROI<DataTypes>::update()
{
    if(m_componentstate == core::objectmodel::ComponentState::Invalid){
        cleanDirty();
        return;
    }

    if(!d_doUpdate.getValue()){
        cleanDirty() ;
        return ;
    }

    const helper::vector<RotatedBox>& rotatedBoxes = m_rotatedBoxes;
    if (rotatedBoxes.empty()) {
        cleanDirty();
        return;
    }

    // Read accessor for input topology
    helper::ReadAccessor< Data<helper::vector<Edge> > > edges = d_edges;
    helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles = d_triangles;
    helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedra = d_tetrahedra;
    helper::ReadAccessor< Data<helper::vector<Hexa> > > hexahedra = d_hexahedra;
    helper::ReadAccessor< Data<helper::vector<Quad> > > quad = d_quad;

    const VecCoord& x0 = d_X0.getValue();

    cleanDirty();

    // Write accessor for topological element indices in BOX
    SetIndex& indices = *d_indices.beginWriteOnly();
    SetIndex& edgeIndices = *d_edgeIndices.beginWriteOnly();
    SetIndex& triangleIndices = *d_triangleIndices.beginWriteOnly();
    SetIndex& tetrahedronIndices = *d_tetrahedronIndices.beginWriteOnly();
    SetIndex& hexahedronIndices = *d_hexahedronIndices.beginWriteOnly();
    SetIndex& quadIndices = *d_quadIndices.beginWriteOnly();

    // Write accessor for toplogical element in BOX
    helper::WriteOnlyAccessor< Data<VecCoord > > pointsInROI = d_pointsInROI;
    helper::WriteOnlyAccessor< Data<helper::vector<Edge> > > edgesInROI = d_edgesInROI;
    helper::WriteOnlyAccessor< Data<helper::vector<Triangle> > > trianglesInROI = d_trianglesInROI;
    helper::WriteOnlyAccessor< Data<helper::vector<Tetra> > > tetrahedraInROI = d_tetrahedraInROI;
    helper::WriteOnlyAccessor< Data<helper::vector<Hexa> > > hexahedraInROI = d_hexahedraInROI;
    helper::WriteOnlyAccessor< Data<helper::vector<Quad> > > quadInROI = d_quadInROI;


    // Clear lists
    indices.clear();
    edgeIndices.clear();
    triangleIndices.clear();
    tetrahedronIndices.clear();
    hexahedronIndices.clear();
    quadIndices.clear();


    pointsInROI.clear();
    edgesInROI.clear();
    trianglesInROI.clear();
    tetrahedraInROI.clear();
    hexahedraInROI.clear();
    quadInROI.clear();


    //Points
    for( unsigned i = 0; i < x0.size(); ++i )
    {
        if (isPointInBoxes(i))
        {
            indices.push_back(i);
            pointsInROI.push_back(x0[i]);
        }
    }

    //Edges
    if (d_computeEdges.getValue())
    {
        for(unsigned int i = 0; i < edges.size(); i++)
        {
            Edge e = edges[i];
            if (isEdgeInBoxes(e))
            {
                edgeIndices.push_back(i);
                edgesInROI.push_back(e);
            }
        }
    }

    //Triangles
    if (d_computeTriangles.getValue())
    {
        for(unsigned int i = 0; i < triangles.size(); i++)
        {
            Triangle t = triangles[i];
            if (isTriangleInBoxes(t))
            {
                triangleIndices.push_back(i);
                trianglesInROI.push_back(t);
            }
        }
    }

    //Tetrahedra
    if (d_computeTetrahedra.getValue())
    {
        for(unsigned int i = 0; i < tetrahedra.size(); i++)
        {
            Tetra t = tetrahedra[i];
            if (isTetrahedronInBoxes(t))
            {
                tetrahedronIndices.push_back(i);
                tetrahedraInROI.push_back(t);
            }
        }
    }

    //Hexahedra
    if (d_computeHexahedra.getValue())
    {
        for(unsigned int i = 0; i < hexahedra.size(); i++)
        {
            Hexa t = hexahedra[i];
            if (isHexahedronInBoxes(t))
            {
                hexahedronIndices.push_back(i);
                hexahedraInROI.push_back(t);
                break;
            }
        }
    }

    //Quads
    if (d_computeQuad.getValue())
    {
        for(unsigned int i = 0; i < quad.size(); i++)
        {
            Quad q = quad[i];
            if (isQuadInBoxes(q))
            {
                quadIndices.push_back(i);
                quadInROI.push_back(q);
            }
        }
    }


    d_nbIndices.setValue(indices.size());

    d_indices.endEdit();
    d_edgeIndices.endEdit();
    d_triangleIndices.endEdit();
    d_tetrahedronIndices.endEdit();
    d_hexahedronIndices.endEdit();
    d_quadIndices.endEdit();

}


template <class DataTypes>
void RotatableBoxROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(m_componentstate == core::objectmodel::ComponentState::Invalid)
        return ;

    if (!vparams->displayFlags().getShowBehaviorModels() && !this->d_drawSize.getValue())
        return;

    const VecCoord& x0 = d_X0.getValue();
    defaulttype::Vec4f color = defaulttype::Vec4f(1.0f, 0.4f, 0.4f, 1.0f);


    ///draw the boxes
    if( d_drawBoxes.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        helper::vector<defaulttype::Vector3> vertices;

        const helper::vector<RotatedBox>& rotatedBoxes = m_rotatedBoxes;

        for (unsigned int bi = 0; bi < rotatedBoxes.size(); ++bi)
        {
            const RotatedBox& box = rotatedBoxes[bi];

            vertices.push_back( box.boxPoints[0] );
            vertices.push_back( box.boxPoints[1] );

            vertices.push_back( box.boxPoints[1] );
            vertices.push_back( box.boxPoints[2] );

            vertices.push_back( box.boxPoints[2] );
            vertices.push_back( box.boxPoints[3] );

            vertices.push_back( box.boxPoints[3] );
            vertices.push_back( box.boxPoints[0] );

            vertices.push_back( box.boxPoints[4] );
            vertices.push_back( box.boxPoints[5] );

            vertices.push_back( box.boxPoints[5] );
            vertices.push_back( box.boxPoints[6] );

            vertices.push_back( box.boxPoints[6] );
            vertices.push_back( box.boxPoints[7] );

            vertices.push_back( box.boxPoints[7] );
            vertices.push_back( box.boxPoints[4] );

            vertices.push_back( box.boxPoints[0] );
            vertices.push_back( box.boxPoints[4] );

            vertices.push_back( box.boxPoints[1] );
            vertices.push_back( box.boxPoints[5] );

            vertices.push_back( box.boxPoints[2] );
            vertices.push_back( box.boxPoints[6] );

            vertices.push_back( box.boxPoints[3] );
            vertices.push_back( box.boxPoints[7] );
            vparams->drawTool()->drawLines(vertices, linesWidth , color );
        }
    }

    const unsigned int max_spatial_dimensions = std::min((unsigned int)3,(unsigned int)DataTypes::spatial_dimensions);

    ///draw points in ROI
    if( d_drawPoints.getValue())
    {
        float pointsWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        vparams->drawTool()->setLightingEnabled(false);
        std::vector<defaulttype::Vector3> vertices;
        helper::ReadAccessor< Data<VecCoord > > pointsInROI = d_pointsInROI;
        for (unsigned int i=0; i<pointsInROI.size() ; ++i)
        {
            CPos p = DataTypes::getCPos(pointsInROI[i]);
            defaulttype::Vector3 pv;
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
        }
        vparams->drawTool()->drawPoints(vertices, pointsWidth, color);
    }

    ///draw edges in ROI
    if( d_drawEdges.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        std::vector<defaulttype::Vector3> vertices;
        helper::ReadAccessor< Data<helper::vector<Edge> > > edgesInROI = d_edgesInROI;
        for (unsigned int i=0; i<edgesInROI.size() ; ++i)
        {
            Edge e = edgesInROI[i];
            for (unsigned int j=0 ; j<2 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[e[j]]);
                defaulttype::Vector3 pv;
                for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                    pv[j] = p[j];
                vertices.push_back( pv );
            }
        }
        vparams->drawTool()->drawLines(vertices, linesWidth, color);
    }

    ///draw triangles in ROI
    if( d_drawTriangles.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        std::vector<defaulttype::Vector3> vertices;
        helper::ReadAccessor< Data<helper::vector<Triangle> > > trianglesInROI = d_trianglesInROI;
        for (unsigned int i=0; i<trianglesInROI.size() ; ++i)
        {
            Triangle t = trianglesInROI[i];
            for (unsigned int j=0 ; j<3 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[t[j]]);
                defaulttype::Vector3 pv;
                for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                    pv[j] = p[j];
                vertices.push_back( pv );
            }
        }
        vparams->drawTool()->drawTriangles(vertices, color);
    }

    ///draw tetrahedra in ROI
    if( d_drawTetrahedra.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        helper::vector<defaulttype::Vector3> vertices;
        helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedraInROI = d_tetrahedraInROI;
        for (unsigned int i=0; i<tetrahedraInROI.size() ; ++i)
        {
            Tetra t = tetrahedraInROI[i];
            for (unsigned int j=0 ; j<4 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[t[j]]);
                defaulttype::Vector3 pv;
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );

                p = DataTypes::getCPos(x0[t[(j+1)%4]]);
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );
            }

            CPos p = DataTypes::getCPos(x0[t[0]]);
            defaulttype::Vector3 pv;
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[2]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[1]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[3]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
        }
        vparams->drawTool()->drawLines(vertices, linesWidth, color);
    }

    ///draw hexahedra in ROI
    if( d_drawHexahedra.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        helper::vector<defaulttype::Vector3> vertices;
        helper::ReadAccessor< Data<helper::vector<Hexa> > > hexahedraInROI = d_hexahedraInROI;
        for (unsigned int i=0; i<hexahedraInROI.size() ; ++i)
        {
            Hexa t = hexahedraInROI[i];
            for (unsigned int j=0 ; j<8 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[t[j]]);
                defaulttype::Vector3 pv;
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );

                p = DataTypes::getCPos(x0[t[(j+1)%4]]);
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );
            }

            CPos p = DataTypes::getCPos(x0[t[0]]);
            defaulttype::Vector3 pv;
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[2]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[1]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[3]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[4]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[5]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[6]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[7]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
        }
        vparams->drawTool()->drawLines(vertices, linesWidth, color);
    }

    ///draw quads in ROI
    if( d_drawQuads.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        helper::vector<defaulttype::Vector3> vertices;
        helper::ReadAccessor<Data<helper::vector<Quad> > > quadsInROI = d_quadInROI;
        for (unsigned i=0; i<quadsInROI.size(); ++i)
        {
            Quad q = quadsInROI[i];
            for (unsigned j=0; j<4; j++)
            {
                CPos p = DataTypes::getCPos(x0[q[j]]);
                defaulttype::Vector3 pv;
                for (unsigned k=0; k<max_spatial_dimensions; k++)
                    pv[k] = p[k];
                vertices.push_back(pv);
            }
            for (unsigned j=0; j<4; j++)
            {
                CPos p = DataTypes::getCPos(x0[q[(j+1)%4]]);
                defaulttype::Vector3 pv;
                for (unsigned k=0; k<max_spatial_dimensions; k++)
                    pv[k] = p[k];
                vertices.push_back(pv);
            }

        }
        vparams->drawTool()->drawLines(vertices,linesWidth,color);
    }

}


template <class DataTypes>
void RotatableBoxROI<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    if( onlyVisible && !d_drawBoxes.getValue() )
        return;

    if(m_componentstate == core::objectmodel::ComponentState::Invalid)
        return ;

    const helper::vector<RotatedBox>& rotatedBoxes = m_rotatedBoxes;

    const Real max_real = std::numeric_limits<Real>::max();
    const Real min_real = std::numeric_limits<Real>::min();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

    for (unsigned int bi = 0; bi < rotatedBoxes.size(); ++bi)
    {
        const helper::vector<Vec3>& boxData = rotatedBoxes[bi].boxPoints;
        for (unsigned int index = 0; index < boxData.size(); index++) {
            if (boxData[index][0] < minBBox[0]) minBBox[0] = boxData[index][0];
            if (boxData[index][1] < minBBox[1]) minBBox[1] = boxData[index][1];
            if (boxData[index][2] < minBBox[2]) minBBox[2] = boxData[index][2];
            if (boxData[index][0] > maxBBox[0]) maxBBox[0] = boxData[index][0];
            if (boxData[index][1] > maxBBox[1]) maxBBox[1] = boxData[index][1];
            if (boxData[index][2] > maxBBox[2]) maxBBox[2] = boxData[index][2];
        }
    }

    this->f_bbox.setValue(params, defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
}


template<class DataTypes>
void RotatableBoxROI<DataTypes>::handleEvent(core::objectmodel::Event *event)
{
    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        setDirtyValue();
        update();
    }
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
