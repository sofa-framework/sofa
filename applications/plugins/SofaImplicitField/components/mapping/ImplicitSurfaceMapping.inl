/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include "ImplicitSurfaceMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/rmath.h>
#include <map>
#include <list>

namespace sofaimplicitfield::mapping
{

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::init()
{
    core::Mapping<In,Out>::init();
    MeshTopology::init();
}

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->Inherit::parse(arg);
    if ( arg->getAttribute("minx") || arg->getAttribute("miny") || arg->getAttribute("minz"))
        this->setGridMin(arg->getAttributeAsFloat("minx",-100.0),
                         arg->getAttributeAsFloat("miny",-100.0),
                         arg->getAttributeAsFloat("minz",-100.0));
    if (arg->getAttribute("maxx") || arg->getAttribute("maxy") || arg->getAttribute("maxz"))
        this->setGridMax(arg->getAttributeAsFloat("maxx",100.0),
                         arg->getAttributeAsFloat("maxy",100.0),
                         arg->getAttributeAsFloat("maxz",100.0));
}

template<class Real>
Real sqr(Real r)
{
    return r*r;
}

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    OutVecCoord &out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();

    InReal invStep = (InReal)(1/mStep.getValue());
    out.resize(0);
    clear();

    if (in.size()==0)
    {
        dOut.endEdit();
        return;
    }

    InReal xmin, xmax;
    InReal ymin, ymax;
    xmin = xmax = in[0][0]*invStep;
    ymin = ymax = in[0][1]*invStep;
    const InReal r = (InReal)(getRadius() / mStep.getValue());
    std::map<int, std::list< InCoord > > sortParticles;
    for (unsigned int ip=0; ip<in.size(); ip++)
    {
        InCoord c0 = in[ip];
        if (c0[0] < (*mGridMin.beginEdit())[0] || c0[0] > (*mGridMax.beginEdit())[0] ||
            c0[1] < (*mGridMin.beginEdit())[1] || c0[1] > (*mGridMax.beginEdit())[1] ||
            c0[2] < (*mGridMin.beginEdit())[2] || c0[2] > (*mGridMax.beginEdit())[2])
            continue;
        InCoord c = c0 * invStep;
        if (c[0] < xmin)
            xmin = c[0];
        else if (c[0] > xmax)
            xmax = c[0];
        if (c[1] < ymin)
            ymin = c[1];
        else if (c[1] > ymax)
            ymax = c[1];
        int z0 = helper::rceil(c[2]-r);
        int z1 = helper::rfloor(c[2]+r);
        for (int z = z0; z < z1; ++z)
            sortParticles[z].push_back(c);
    }

    OutReal r2 = (OutReal)sqr(r);
    type::BoundingBox bigBox {mGridMin.getValue(), mGridMax.getValue()};
    type::BoundingBox box;
    for(auto& [_, z_plane] : sortParticles)
    {
        for(auto& particle : z_plane)
            box.include(particle);
    }
    box.intersection(bigBox);

    auto fieldFunction = [&sortParticles, &r, &r2](Vec3d& pos) -> double {
        int index = helper::rceil(pos.z());

        double sumd = 0.0;
        for(auto& particle : sortParticles[index]){
            double d2 = (pos - particle).norm2();
            if(d2 < r2){
                d2 /= r2;
                sumd += (1 + (-4*d2*d2*d2 + 17*d2*d2 - 22*d2)/9);
            }
        }
        return sumd;
    };

    auto triangles = helper::getWriteOnlyAccessor(d_seqTriangles);
    auto points = helper::getWriteOnlyAccessor(dOut);

    points.clear();
    triangles.clear();
    marchingCube.generateSurfaceMesh(mIsoValue.getValue(), mStep.getValue(),
                                     invStep, box.minBBox(), box.maxBBox(),
                                     fieldFunction, points.wref(), triangles.wref());


}

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::applyJ(const core::MechanicalParams * /*mparams*/, Data<OutVecDeriv>& /*dOut*/, const Data<InVecDeriv>& /*dIn*/)
{
}

}
