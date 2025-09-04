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
void ImplicitSurfaceMapping<In,Out>::draw(const core::visual::VisualParams* params)
{
    auto dt = params->drawTool();

    dt->drawBoundingBox(mGridMin.getValue(), mGridMax.getValue());
    dt->drawBoundingBox(mLocalGridMin, mLocalGridMax);
}

template <class In, class Out>
void ImplicitSurfaceMapping<In,Out>::apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    const InVecCoord& in = dIn.getValue();

    clear();

    if (in.size()==0)
    {
        OutVecCoord &out = *dOut.beginEdit();
        dOut.endEdit();
        return;
    }

    auto minGrid = mGridMin.getValue();
    auto maxGrid = mGridMax.getValue();

    InReal invStep = (InReal)(1/mStep.getValue());
    const InReal r = getRadius();

    std::unordered_map<int, std::list< InCoord > > sortParticles;
    for (unsigned int ip=0; ip<in.size(); ip++)
    {
        InCoord c0 = in[ip];
        if (c0[0] < minGrid[0] || c0[0] > maxGrid[0] ||
            c0[1] < minGrid[1] || c0[1] > maxGrid[1] ||
            c0[2] < minGrid[2] || c0[2] > maxGrid[2])
            continue;

        InCoord c = c0 ;
        int z0 = helper::rfloor((c[2]-r)*invStep);
        int z1 = helper::rceil((c[2]+r)*invStep);
        for (int z = z0; z <= z1; ++z)
            sortParticles[z].push_back(c);
    }

    OutReal r2 = (OutReal)sqr(r);

    double rr = getRadius();
    type::BoundingBox box{};
    for(auto& particle : in)
    {
        box.include(particle);
    }
    box.include(box.minBBox()+Vec3d{-rr,-rr,-rr});
    box.include(box.maxBBox()+Vec3d{+rr,+rr,+rr});

    mLocalGridMin = box.minBBox();
    mLocalGridMax = box.maxBBox();

    type::BoundingBox bigBox {mGridMin.getValue(), mGridMax.getValue()};
    box.intersection(bigBox);

    auto fieldFunction = [&sortParticles, &r, &r2, &invStep](
                         std::vector<Vec3d>& pos, std::vector<double>& res) -> void {

        auto z = pos[0].z();
        int index = helper::rfloor(z*invStep);
        auto particlesIt = sortParticles.find(index);
        if(particlesIt==sortParticles.end())
            return;

        int i = 0;
        for(auto& position : pos )
        {
            double sumd = 0.0;
            for(auto& particle : (particlesIt->second)){
                position.z() = z;
                double d2 = (position - particle).norm2();
                if(d2 < r2){
                    d2 /= r2;
                    sumd += (1 + (-4*d2*d2*d2 + 17*d2*d2 - 22*d2)/9);
                }
            }
            res[i++] = sumd;
        }
        return;
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
