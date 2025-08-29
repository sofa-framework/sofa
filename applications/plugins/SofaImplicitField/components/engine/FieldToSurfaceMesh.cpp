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
    , mStep(initData(&mStep,0.1,"step","Step"))
    , mIsoValue(initData(&mIsoValue,0.0,"isoValue","Iso Value"))
    , mGridMin(initData(&mGridMin, Vec3d(-1,-1,-1),"min","Grid Min"))
    , mGridMax(initData(&mGridMax, Vec3d(1,1,1),"max","Grid Max"))
    , d_outPoints(initData(&d_outPoints, "points", "position of the tiangles vertex"))
    , d_outTriangles(initData(&d_outTriangles, "triangles", "list of triangles"))
    , d_debugDraw(initData(&d_debugDraw,false, "debugDraw","Display the extracted surface"))
{
    addUpdateCallback("updateMesh", {&mStep, &mIsoValue, &mGridMin, &mGridMax}, [this](const sofa::core::DataTracker&)
    {
        checkInputs();
        hasChanged=true;
        return core::objectmodel::ComponentState::Valid;
    }, {});
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

    updateMeshIfNeeded();

    d_componentState = core::objectmodel::ComponentState::Valid;
}

void FieldToSurfaceMesh::checkInputs(){

    auto length = mGridMax.getValue()-mGridMin.getValue() ;
    auto step = mStep.getValue();

    // clamp the mStep value to avoid too large grids
    if( step < 0.0001 || (length.x() / step > 256) || length.y() / step > 256 || length.z() / step > 256)
    {
        mStep.setValue( *std::max_element(length.begin(), length.end()) / 256.0 );
        msg_warning() << "step exceeding grid size, clamped to " << mStep.getValue();
    }
}

void FieldToSurfaceMesh::updateMeshIfNeeded()
{
    if(!hasChanged)
        return;

    sofa::helper::getWriteOnlyAccessor(d_outPoints).clear();
    sofa::helper::getWriteOnlyAccessor(d_outTriangles).clear();

    double isoval = mIsoValue.getValue();
    double mstep = mStep.getValue();
    double invStep = 1.0/mStep.getValue();

    Vec3d gridmin = mGridMin.getValue() ;
    Vec3d gridmax = mGridMax.getValue() ;

    auto field = l_field.get();

    generateSurfaceMesh(isoval, mstep, invStep, gridmin, gridmax, field);

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

    updateMeshIfNeeded();

    auto drawTool = vparams->drawTool();

    drawTool->drawBoundingBox(mGridMin.getValue(), mGridMax.getValue()) ;

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

        Vec3d a1 = x[c]-x[b] ;
        Vec3d a2 = x[a]-x[b] ;

        vparams->drawTool()->drawTriangles({pa,pb,pc},
                                           a1.cross(a2),
                                           type::RGBAColor(0.0,0.0,1.0,1.0));
    }

    if(x.size()>1000){
        drawTool->drawPoints(x, 1.0, type::RGBAColor(1.0,1.0,0.0,0.2)) ;
    }else{
        drawTool->drawSpheres(x, 0.01, type::RGBAColor(1.0,1.0,0.0,0.2)) ;
    }
}

void FieldToSurfaceMesh::generateSurfaceMesh(double isoval, double mstep, double invStep,
                                             Vec3d gridmin, Vec3d gridmax,
                                             sofa::component::geometry::ScalarField* field)
{
    if(!field)
        return;

    tmpPoints.clear();
    tmpTriangles.clear();

    int nx = floor((gridmax.x() - gridmin.x()) * invStep) + 1 ;
    int ny = floor((gridmax.y() - gridmin.y()) * invStep) + 1 ;
    int nz = floor((gridmax.z() - gridmin.z()) * invStep) + 1 ;

    double cx,cy,cz;
    int x,y,z,i,mk;
    const int *tri;


    planes.resize(2*(nx)*(ny));
    P0 = planes.begin()+0;
    P1 = planes.begin()+nx*ny;

    const int dx = 1;
    const int dy = nx;

    z = 0;
    newPlane();

    i = 0 ;
    cz = gridmin.z()  ;
    for (int y = 0 ; y < ny ; ++y)
    {
        cy = gridmin.y() + mstep * y ;
        for (int x = 0 ; x < nx ; ++x, ++i)
        {
            cx = gridmin.x() + mstep * x ;

            Vec3d pos { cx, cy, cz }  ;
            double res = field->getValue(pos) ;
            (P1+i)->data = res ;
        }
    }

    for (z=1; z<=nz; ++z)
    {
        newPlane();

        i = 0 ;
        cz = gridmin.z() + mstep * z ;
        for (int y = 0 ; y < ny ; ++y)
        {
            cy = gridmin.y() + mstep * y ;
            for (int x = 0 ; x < nx ; ++x, ++i)
            {
                cx = gridmin.x() + mstep * x ;

                Vec3d pos { cx, cy, cz }  ;
                double res = field->getValue(pos) ;
                (P1+i)->data = res ;
            }
        }

        unsigned int i=0;
        int edgecube[12];
        const int edgepts[12] = {0,1,0,1,0,1,0,1,2,2,2,2};
        typename std::vector<CubeData>::iterator base = planes.begin();
        int ip0 = P0-base;
        int ip1 = P1-base;
        edgecube[0]  = (ip0   -dy);
        edgecube[1]  = (ip0      );
        edgecube[2]  = (ip0      );
        edgecube[3]  = (ip0-dx   );
        edgecube[4]  = (ip1   -dy);
        edgecube[5]  = (ip1      );
        edgecube[6]  = (ip1      );
        edgecube[7]  = (ip1-dx   );
        edgecube[8]  = (ip1-dx-dy);
        edgecube[9]  = (ip1-dy   );
        edgecube[10] = (ip1      );
        edgecube[11] = (ip1-dx   );

        // First line is all zero
        {
            y=0;
            x=0;
            i+=nx;
        }
        for(y=1; y<ny; y++)
        {
            // First column is all zero
            x=0;
            ++i;

            for(x=1; x<nx; x++)
            {
                Vec3d pos(x, y, z);
                if (((P1+i)->data>isoval)^((P1+i-dx)->data>isoval))
                {
                    (P1+i)->p[0] = addPoint(tmpPoints, 0, pos,gridmin, (P1+i)->data,(P1+i-dx)->data, mstep, isoval);
                }
                if (((P1+i)->data>isoval)^((P1+i-dy)->data>isoval))
                {
                    (P1+i)->p[1] = addPoint(tmpPoints, 1, pos,gridmin,(P1+i)->data,(P1+i-dy)->data, mstep, isoval);
                }
                if (((P1+i)->data>isoval)^((P0+i)->data>isoval))
                {
                    (P1+i)->p[2] = addPoint(tmpPoints, 2, pos,gridmin,(P1+i)->data,(P0+i)->data, mstep, isoval);
                }

                // All points should now be created
                if ((P0+i-dx-dy)->data > isoval) mk = 1;
                else mk=0;
                if ((P0+i   -dy)->data > isoval) mk|= 2;
                if ((P0+i      )->data > isoval) mk|= 4;
                if ((P0+i-dx   )->data > isoval) mk|= 8;
                if ((P1+i-dx-dy)->data > isoval) mk|= 16;
                if ((P1+i   -dy)->data > isoval) mk|= 32;
                if ((P1+i      )->data > isoval) mk|= 64;
                if ((P1+i-dx   )->data > isoval) mk|= 128;

                tri=sofa::helper::MarchingCubeTriTable[mk];
                while (*tri>=0)
                {
                    typename std::vector<CubeData>::iterator b = base+i;
                    addFace(tmpTriangles,
                                (b+edgecube[tri[0]])->p[edgepts[tri[0]]],
                                (b+edgecube[tri[1]])->p[edgepts[tri[1]]],
                                (b+edgecube[tri[2]])->p[edgepts[tri[2]]], tmpPoints.size());
                    tri+=3;
                }
                ++i;
            }
        }
    }
}

void FieldToSurfaceMesh::newPlane()
{
    CubeData c;
    c.p[0] = -1;
    c.p[1] = -1;
    c.p[2] = -1;
    c.data = 0;
    typename std::vector<CubeData>::iterator P = P0;
    P0 = P1;
    P1 = P;
    int n = planes.size()/2;
    for (int i=0; i<n; ++i,++P)
        *P = c;
}

// Register in the Factory
void registerFieldToSurfaceMesh(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Generates a surface mesh from a field function.")
                             .add< FieldToSurfaceMesh >());
}

}
