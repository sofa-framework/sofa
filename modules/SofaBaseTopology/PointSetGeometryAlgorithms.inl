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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_INL

#include <SofaBaseTopology/PointSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>

namespace sofa
{

namespace component
{

namespace topology
{

template <class DataTypes>
void PointSetGeometryAlgorithms< DataTypes >::init()
{
    object = this->getContext()->core::objectmodel::BaseContext::template get< core::behavior::MechanicalState< DataTypes > >();
    core::topology::GeometryAlgorithms::init();
    this->m_topology = this->getContext()->getMeshTopology();
}

template <class DataTypes>
void PointSetGeometryAlgorithms< DataTypes >::reinit()
{
}

template <class DataTypes>
float PointSetGeometryAlgorithms< DataTypes >::getIndicesScale() const
{
    const sofa::defaulttype::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
    return (float)((bbox.maxBBox() - bbox.minBBox()).norm() * showIndicesScale.getValue());
}


template <class DataTypes>
typename DataTypes::Coord PointSetGeometryAlgorithms<DataTypes>::getPointSetCenter() const
{
    typename DataTypes::Coord center;
    // get current positions
    const typename DataTypes::VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());

    const int numVertices = this->m_topology->getNbPoints();
    for(int i=0; i<numVertices; ++i)
    {
        center += p[i];
    }

    center /= numVertices;
    return center;
}

template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getEnclosingSphere(typename DataTypes::Coord &center,
        typename DataTypes::Real &radius) const
{
    // get current positions
    const typename DataTypes::VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());

    const unsigned int numVertices = this->m_topology->getNbPoints();
    for(unsigned int i=0; i<numVertices; ++i)
    {
        center += p[i];
    }
    center /= numVertices;
    radius = (Real) 0;

    for(unsigned int i=0; i<numVertices; ++i)
    {
        const CPos dp = DataTypes::getCPos(center)-DataTypes::getCPos(p[i]);
        const Real val = dot(dp,dp);
        if(val > radius)
            radius = val;
    }
    radius = (Real)sqrt((double) radius);
}

template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getAABB(typename DataTypes::Real bb[6] ) const
{
    CPos minCoord, maxCoord;
    getAABB(minCoord, maxCoord);

    bb[0] = (NC>0) ? minCoord[0] : (Real)0;
    bb[1] = (NC>1) ? minCoord[1] : (Real)0;
    bb[2] = (NC>2) ? minCoord[2] : (Real)0;
    bb[3] = (NC>0) ? maxCoord[0] : (Real)0;
    bb[4] = (NC>1) ? maxCoord[1] : (Real)0;
    bb[5] = (NC>2) ? maxCoord[2] : (Real)0;
}

template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::getAABB(CPos& minCoord, CPos& maxCoord) const
{
    // get current positions
    const VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());

    minCoord = DataTypes::getCPos(p[0]);
    maxCoord = minCoord;

    for(unsigned int i=1; i<p.size(); ++i)
    {
        CPos pi = DataTypes::getCPos(p[i]);
        for (unsigned int c=0; c<pi.size(); ++c)
            if(minCoord[c] > pi[c]) minCoord[c] = pi[c];
            else if(maxCoord[c] < pi[c]) maxCoord[c] = pi[c];
    }
}

template<class DataTypes>
const typename DataTypes::Coord& PointSetGeometryAlgorithms<DataTypes>::getPointPosition(const PointID pointId) const
{
    // get current positions
    const typename DataTypes::VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());

    return p[pointId];
}

template<class DataTypes>
const typename DataTypes::Coord& PointSetGeometryAlgorithms<DataTypes>::getPointRestPosition(const PointID pointId) const
{
    // get rest positions
    const typename DataTypes::VecCoord& p = (object->read(core::ConstVecCoordId::restPosition())->getValue());

    return p[pointId];
}

template<class DataTypes>
typename PointSetGeometryAlgorithms<DataTypes>::Angle
PointSetGeometryAlgorithms<DataTypes>::computeAngle(PointID ind_p0, PointID ind_p1, PointID ind_p2) const
{
    const double ZERO = 1e-10;
    const typename DataTypes::VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());
    Coord p0 = p[ind_p0];
    Coord p1 = p[ind_p1];
    Coord p2 = p[ind_p2];
    double t = (p1 - p0)*(p2 - p0);

    if(fabs(t) < ZERO)
        return RIGHT;
    if(t > 0.0)
        return ACUTE;
    else
        return OBTUSE;
}


template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::initPointsAdded(const helper::vector< unsigned int > &indices, const helper::vector< core::topology::PointAncestorElem > &ancestorElems
    , const helper::vector< core::VecCoordId >& coordVecs, const helper::vector< core::VecDerivId >& derivVecs )
{
    using namespace sofa::core::topology;

    helper::vector< VecCoord* > pointsAddedVecCoords;
    helper::vector< VecDeriv* > pointsAddedVecDerivs;

    const unsigned int nbPointCoords = coordVecs.size();
    const unsigned int nbPointDerivs = derivVecs.size();

    for (unsigned int i=0; i < nbPointCoords; i++)
    {
        pointsAddedVecCoords.push_back(this->object->write(coordVecs[i])->beginEdit());
    }

    for (unsigned int i=0; i < nbPointDerivs; i++)
    {
        pointsAddedVecDerivs.push_back(this->object->write(derivVecs[i])->beginEdit());
    }

    for (unsigned int i=0; i < indices.size(); i++)
    {
        if (ancestorElems[i].index != BaseMeshTopology::InvalidID)
        {
            initPointAdded(indices[i], ancestorElems[i], pointsAddedVecCoords, pointsAddedVecDerivs);
        }
    }

    for (unsigned int i=0; i < nbPointCoords; i++)
    {
        this->object->write(coordVecs[i])->endEdit();
    }

    for (unsigned int i=0; i < nbPointDerivs; i++)
    {
        this->object->write(derivVecs[i])->endEdit();
    }
}


template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::initPointAdded(unsigned int index, const core::topology::PointAncestorElem &ancestorElem
    , const helper::vector< VecCoord* >& coordVecs, const helper::vector< VecDeriv* >& /*derivVecs*/)

{
    using namespace sofa::core::topology;

    for (unsigned int i = 0; i < coordVecs.size(); i++)
    {
        (*coordVecs[i])[index] = (*coordVecs[i])[ancestorElem.index];
        DataTypes::add((*coordVecs[i])[index], ancestorElem.localCoords[0], ancestorElem.localCoords[1], ancestorElem.localCoords[2]);
    }
}


template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* )
{
#ifndef SOFA_NO_OPENGL
    if (showPointIndices.getValue())
    {
        sofa::defaulttype::Mat<4,4, GLfloat> modelviewM;
        sofa::defaulttype::Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());

        sofa::simulation::Node* context = dynamic_cast<sofa::simulation::Node*>(this->getContext());
        glColor3f(1.0,1.0,1.0);
        glDisable(GL_LIGHTING);
        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)context, sceneMinBBox.ptr(), sceneMaxBBox.ptr());

        float PointIndicesScale = getIndicesScale();
        //float scale = showIndicesScale.getValue();

        for (unsigned int i =0; i<coords.size(); i++)
        {
            std::ostringstream oss;
            oss << i;
            std::string tmp = oss.str();
            const char* s = tmp.c_str();
            glPushMatrix();
            sofa::defaulttype::Vec3f center; center = DataTypes::getCPos(coords[i]);
            glTranslatef(center[0], center[1], center[2]);
            glScalef(PointIndicesScale,PointIndicesScale,PointIndicesScale);

            // Makes text always face the viewer by removing the scene rotation
            // get the current modelview matrix
            glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
            modelviewM.transpose();

            sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

            //glLoadMatrixf(modelview);
            glLoadIdentity();

            glTranslatef(temp[0], temp[1], temp[2]);
            glScalef(PointIndicesScale,PointIndicesScale,PointIndicesScale);

            while(*s)
            {
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                s++;
            }

            glPopMatrix();

        }
    }
#endif /* SOFA_NO_OPENGL */
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif
