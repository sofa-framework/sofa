/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2025 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaImplicitField/config.h>

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include <sofa/helper/MarchingCubeUtility.h>

#include "FieldToSurfaceMesh.h"

namespace sofaimplicitfield::component::engine
{

FieldToSurfaceMesh::FieldToSurfaceMesh()
    : l_field(initLink("field", "The scalar field to generate a mesh from."))
    , d_step(initData(&d_step,0.1,"step","Step"))
    , d_IsoValue(initData(&d_IsoValue,0.0,"isoValue","Iso Value"))
    , d_gridMin(initData(&d_gridMin, Vec3d(-1,-1,-1),"min","Grid Min"))
    , d_gridMax(initData(&d_gridMax, Vec3d(1,1,1),"max","Grid Max"))
    , d_outPoints(initData(&d_outPoints, "points", "position of the tiangles vertex"))
    , d_outTriangles(initData(&d_outTriangles, "triangles", "list of triangles"))
    , d_debugDraw(initData(&d_debugDraw,false, "debugDraw","Display the extracted surface"))
{
    addUpdateCallback("updateMesh", {&d_step, &d_IsoValue, &d_gridMin, &d_gridMax}, [this](const sofa::core::DataTracker&)
    {
        checkInputs();
        updateMeshIfNeeded();
        return core::objectmodel::ComponentState::Valid;
    }, {&d_outPoints, &d_outTriangles});
    d_outPoints.setGroup("Output");
    d_outTriangles.setGroup("Output");
}

FieldToSurfaceMesh::~FieldToSurfaceMesh()
{
}

void FieldToSurfaceMesh::init()
{
    if(!l_field.get())
    {
        msg_error() << "Missing field to extract surface from";
        d_componentState = core::objectmodel::ComponentState::Invalid;
    }

    d_componentState = core::objectmodel::ComponentState::Valid;
}

void FieldToSurfaceMesh::computeBBox(const core::ExecParams* /* params */, bool /*onlyVisible*/)
{
    f_bbox.setValue({d_gridMin.getValue(), d_gridMax.getValue()});
}

void FieldToSurfaceMesh::checkInputs(){

    auto length = d_gridMax.getValue()-d_gridMin.getValue() ;
    auto step = d_step.getValue();

    // clamp the mStep value to avoid too large grids
    if( step < 0.0001 || (length.x() / step > 256) || length.y() / step > 256 || length.z() / step > 256)
    {
        d_step.setValue( *std::max_element(length.begin(), length.end()) / 256.0 );
        msg_warning() << "step exceeding grid size, clamped to " << d_step.getValue();
    }
}

void FieldToSurfaceMesh::updateMeshIfNeeded()
{
    sofa::helper::getWriteOnlyAccessor(d_outPoints).clear();
    sofa::helper::getWriteOnlyAccessor(d_outTriangles).clear();

    double isoval = d_IsoValue.getValue();
    double mstep = d_step.getValue();
    double invStep = 1.0/d_step.getValue();

    Vec3d gridmin = d_gridMin.getValue() ;
    Vec3d gridmax = d_gridMax.getValue() ;

    auto field = l_field.get();

    if(!field)
        return;

    // Clear the previously used buffer
    tmpPoints.clear();
    tmpTriangles.clear();

    marchingCube.generateSurfaceMesh(isoval, mstep, invStep, gridmin, gridmax,
                                     [field](std::vector<Vec3d>& positions, std::vector<double>& res){
                                        int i=0;
                                        for(auto& position : positions)
                                        {
                                            res[i++]=field->getValue(position);
                                        }
                                      },
                                     tmpPoints, tmpTriangles);

    /// Copy the surface to Sofa topology
    d_outPoints.setValue(tmpPoints);
    d_outTriangles.setValue(tmpTriangles);

    tmpPoints.clear();
    tmpTriangles.clear();

    hasChanged = false;
    return;
}

void FieldToSurfaceMesh::draw(const VisualParams* vparams)
{
    if(isComponentStateInvalid())
        return;

    if(!d_debugDraw.getValue())
        return;

    auto drawTool = vparams->drawTool();

    sofa::helper::ReadAccessor< Data<VecCoord> > x = d_outPoints;
    sofa::helper::ReadAccessor< Data<SeqTriangles> > triangles = d_outTriangles;
    drawTool->setLightingEnabled(true);

    for(const Triangle& triangle : triangles)
    {
        int a = triangle[0];
        int b = triangle[1];
        int c = triangle[2];
        Vec3d center = (x[a]+x[b]+x[c])*0.333333;
        Vec3d pa = (0.9*x[a]+0.1*center) ;
        Vec3d pb = (0.9*x[b]+0.1*center) ;
        Vec3d pc = (0.9*x[c]+0.1*center) ;

        vparams->drawTool()->drawTriangles({pb,pa,pc},
                                           type::RGBAColor(0.0,0.0,1.0,1.0));
    }

    if(x.size()>1000){
        drawTool->drawPoints(x, 1.0, type::RGBAColor(1.0,1.0,0.0,0.2)) ;
    }else{
        drawTool->drawSpheres(x, 0.01, type::RGBAColor(1.0,1.0,0.0,0.2)) ;
    }
}

// Register in the Factory
void registerFieldToSurfaceMesh(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Generates a surface mesh from a field function.")
                             .add< FieldToSurfaceMesh >());
}

}
