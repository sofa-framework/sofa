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
#include <sofa/component/engine/generate/Spiral.h>
#include <sofa/core/visual/VisualParams.h>

#ifndef M_PI_2
#define M_PI_2 1.570796326794897f
#endif
namespace sofa::component::engine::generate
{

template <class DataTypes>
Spiral<DataTypes>::Spiral()
    : f_X0( initData (&f_X0, "rest_position", "Rest position coordinates of the degrees of freedom") )
    , f_X( initData (&f_X, "position", "Position coordinates of the degrees of freedom") )
    , curvature( initData (&curvature, Real(0.2),"curvature", "Spiral curvature factor") )
{
    addInput(&f_X0);
    addOutput(&f_X);
}

template <class DataTypes>
void Spiral<DataTypes>::init()
{
    setDirtyValue();
}

template <class DataTypes>
void Spiral<DataTypes>::reinit()
{
    update();
}



template <class DataTypes>
void Spiral<DataTypes>::doUpdate()
{
    const VecCoord x0 = f_X0.getValue();

    VecCoord* x = f_X.beginWriteOnly();
    x->clear();

    for( unsigned i=0; i<x0.size(); ++i )
    {
        Real t0 = x0[i].x();

        // x = a t cos t, y = a t sin t
        // l'(t) = a sqrt(1+t²)
        // l = a/2 (t sqrt(1+t²) + asinh(t))
        // l = a/2 (t sqrt(1+t²) + asinh(t))
        Real A = curvature.getValue();
        Real l = (t0 > 0 ? t0 : -t0);
        Real t = sqrt(l);
        // t = t - l(t)/l'(t)
        for (int n=0; n<10; ++n)
        {
#if !defined(WIN32)
            Real l_t = A/2 * ( t * sqrt(1+t*t) + asinh(t) );
#else
            Real l_t = A/2 * ( t * sqrt(1+t*t) + log(t + sqrt(t * t + 1)));
#endif

            Real dl_t = A * sqrt(1+t*t);
            t = t - (l_t - l) / dl_t;
        }
        Real a = t;
        Real r;
        if (t0 > 0) { r = A*t; }
        else        { r = -A*t; }

        sofa::type::Vec<3, Real> v(r*cos(a), r*sin(a), x0[i].z());
        sofa::type::Vec<3, Real> n(- A*sin(t) - A*t*cos(t), A*cos(t) - A*t*sin(t), 0);
        n.normalize();
        if (a > - M_PI/2 && a < M_PI_2 && n.y() > 0.8f)
        {
            n.y() = 0.8f;
            n.x() = -sqrt(1-n.y()*n.y());
        }
        v += n * x0[i].y();
        x->push_back(v);
    }

    f_X.endEdit();
}

template <class DataTypes>
void Spiral<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
}

} //namespace sofa::component::engine::generate
