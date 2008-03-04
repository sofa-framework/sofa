/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_INL
#define SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_INL

#include <sofa/component/mapping/BeamLinearMapping.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <string>



namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

template <class BasicMapping>
void BeamLinearMapping<BasicMapping>::init()
{
    if (this->points.empty() && this->toModel!=NULL)
    {
        VecCoord& x = *this->toModel->getX();
        std::cout << "BeamLinearMapping: init "<<x.size()<<" points."<<std::endl;
        points.resize(x.size());
        for (unsigned int i=0; i<x.size(); i++)
            points[i] = x[i];
    }
    this->BasicMapping::init();
}

template <class BasicMapping>
void BeamLinearMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    //translation = in[index.getValue()].getCenter();
    //in[index.getValue()].writeRotationMatrix(rotation);
    rotatedPoints0.resize(points.size());
    rotatedPoints1.resize(points.size());
    out.resize(points.size());
    for(unsigned int i=0; i<points.size(); i++)
    {
        Coord inpos = points[i];
        int in0 = helper::rfloor(inpos[0]);
        if (in0<0) in0 = 0; else if (in0 > (int)in.size()-2) in0 = in.size()-2;
        inpos[0] -= in0;
        rotatedPoints0[i] = in[in0].getOrientation().rotate(inpos);
        Coord out0 = in[in0].getCenter() + rotatedPoints0[i];
        Coord inpos1 = inpos; inpos1[0] -= 1;
        rotatedPoints1[i] = in[in0+1].getOrientation().rotate(inpos1);
        Coord out1 = in[in0+1].getCenter() + rotatedPoints1[i];
        Real fact = (Real)inpos[0];
        fact = 3*(fact*fact)-2*(fact*fact*fact);
        out[i] = out0 * (1-fact) + out1 * (fact);
    }
}

template <class BasicMapping>
void BeamLinearMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    //const typename In::VecCoord& x = *this->fromModel->getX();
    //Deriv v,omega;
    //v = in[index.getValue()].getVCenter();
    //omega = in[index.getValue()].getVOrientation();
    out.resize(points.size());
    for(unsigned int i=0; i<points.size(); i++)
    {
        // out = J in
        // J = [ I -OM^ ]
        //out[i] =  v - cross(rotatedPoints[i],omega);

        defaulttype::Vec<N, typename In::Real> inpos = points[i];
        int in0 = helper::rfloor(inpos[0]);
        if (in0<0) in0 = 0; else if (in0 > (int)in.size()-2) in0 = in.size()-2;
        inpos[0] -= in0;
        Deriv omega0 = in[in0].getVOrientation();
        Deriv out0 = in[in0].getVCenter() - cross(rotatedPoints0[i], omega0);
        Deriv omega1 = in[in0+1].getVOrientation();
        Deriv out1 = in[in0+1].getVCenter() - cross(rotatedPoints1[i], omega1);
        Real fact = (Real)inpos[0];
        fact = 3*(fact*fact)-2*(fact*fact*fact);
        out[i] = out0 * (1-fact) + out1 * (fact);
    }
}

/// Template specialization for 2D rigids

// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <class BasicMapping>
void BeamLinearMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    //Deriv v,omega;
    for(unsigned int i=0; i<points.size(); i++)
    {
        // out = Jt in
        // Jt = [ I     ]
        //      [ -OM^t ]
        // -OM^t = OM^

        //Deriv f = in[i];
        //v += f;
        //omega += cross(rotatedPoints[i],f);

        defaulttype::Vec<N, typename In::Real> inpos = points[i];
        int in0 = helper::rfloor(inpos[0]);
        if (in0<0) in0 = 0; else if (in0 > (int)out.size()-2) in0 = out.size()-2;
        inpos[0] -= in0;
        Deriv f = in[i];
        Real fact = (Real)inpos[0];
        fact = 3*(fact*fact)-2*(fact*fact*fact);
        out[in0].getVCenter() += f * (1-fact);
        out[in0].getVOrientation() += cross(rotatedPoints0[i], f) * (1-fact);
        out[in0+1].getVCenter() += f * (fact);
        out[in0+1].getVOrientation() += cross(rotatedPoints1[i], f) * (fact);
    }
    //out[index.getValue()].getVCenter() += v;
    //out[index.getValue()].getVOrientation() += omega;
}


/// Template specialization for 2D rigids

// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );


// BeamLinearMapping::applyJT( typename In::VecConst& out, const typename Out::VecConst& in ) //
// this function propagate the constraint through the rigid mapping :
// if one constraint along (vector n) with a value (v) is applied on the childModel (like collision model)
// then this constraint is transformed by (Jt.n) with value (v) for the rigid model
// There is a specificity of this propagateConstraint: we have to find the application point on the childModel
// in order to compute the right constaint on the rigidModel.
template <class BasicMapping>
void BeamLinearMapping<BasicMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    const typename In::VecCoord& x = *this->fromModel->getX();
    int outSize = out.size();
    out.resize(in.size() + outSize); // we can accumulate in "out" constraints from several mappings

    for(unsigned int i=0; i<in.size(); i++)
    {
        // computation of (Jt.n) //
        // in[i].size() = num node involved in the constraint
        for (unsigned int j=0; j<in[i].size(); j++)
        {
            int index = in[i][j].index;	// index of the node
            // interpolation//////////
            defaulttype::Vec<N, typename In::Real> inpos = points[index];
            int in0 = helper::rfloor(inpos[0]);
            if (in0<0) in0 = 0; else if (in0 > (int)x.size()-2) in0 = x.size()-2;
            inpos[0] -= in0;
            Real fact = (Real)inpos[0];
            fact = 3*(fact*fact)-2*(fact*fact*fact);
            /////////////////////////
            Deriv w_n = (Deriv) in[i][j].data;	// weighted value of the constraint direction

            // Compute the mapped Constraint on the beam nodes ///
            InDeriv direction0;
            direction0.getVCenter() = w_n * (1-fact);
            direction0.getVOrientation() = cross(rotatedPoints0[index], w_n) * (1-fact);
            InDeriv direction1;
            direction1.getVCenter() = w_n * (fact);
            direction1.getVOrientation() = cross(rotatedPoints1[index], w_n) * (fact);
            out[outSize+i].push_back(InSparseDeriv(in0, direction0));
            out[outSize+i].push_back(InSparseDeriv(in0+1, direction1));
        }

    }
}
/// Template specialization for 2D rigids

// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
// template<>
// void BeamLinearMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );


template <class BasicMapping>
void BeamLinearMapping<BasicMapping>::draw()
{
    if (!this->getShow()) return;
    glDisable (GL_LIGHTING);
    glPointSize(7);
    glColor4f (1,1,0,1);
    glBegin (GL_POINTS);
    const typename Out::VecCoord& x = *this->toModel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::glVertexT(x[i]);
    }
    glEnd();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
