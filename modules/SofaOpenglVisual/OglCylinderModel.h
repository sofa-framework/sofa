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
#ifndef SOFA_COMPONENT_VISUALMODEL_POINTSPLATMODEL_H
#define SOFA_COMPONENT_VISUALMODEL_POINTSPLATMODEL_H

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/SofaGeneral.h>

namespace sofa
{
namespace core
{
namespace topology
{
class BaseMeshTopology;
}
namespace behavior
{
class BaseMechanicalState;
}
}

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API OglCylinderModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglCylinderModel,core::visual::VisualModel);
protected:
    OglCylinderModel();
    virtual ~OglCylinderModel();
public:
    virtual void init();

    virtual void reinit();


    virtual void draw(const core::visual::VisualParams* vparams);

private:
    void setColor(float r, float g, float b, float a);
    void setColor(std::string color);

private:
    Data<float>		radius;
    // Data<float>		alpha;
    Data<std::string>	color;

    core::topology::BaseMeshTopology*	_topology;
    core::behavior::BaseMechanicalState* _mstate;

    float r,g,b,a;
    // component::topology::PointData<sofa::helper::vector<unsigned char> >		pointData;

    typedef defaulttype::ExtVec3fTypes::Coord Coord;
    typedef defaulttype::ExtVec3fTypes::VecCoord VecCoord;
    typedef defaulttype::ExtVec3fTypes::Real Real;
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
