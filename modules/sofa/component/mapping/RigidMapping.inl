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
#ifndef SOFA_COMPONENT_MAPPING_RIGIDMAPPING_INL
#define SOFA_COMPONENT_MAPPING_RIGIDMAPPING_INL

#include <sofa/component/mapping/RigidMapping.h>
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
class RigidMapping<BasicMapping>::Loader : public helper::io::MassSpringLoader, public helper::io::SphereLoader
{
public:
    RigidMapping<BasicMapping>* dest;
    Loader(RigidMapping<BasicMapping>* dest) : dest(dest) {}
    virtual void addMass(double px, double py, double pz, double, double, double, double, double, bool, bool)
    {
        Coord c;
        Out::DataTypes::set(c,px,py,pz);
        dest->points.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
    }
    virtual void addSphere(double px, double py, double pz, double)
    {
        Coord c;
        Out::DataTypes::set(c,px,py,pz);
        dest->points.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
    }
};

template <class BasicMapping>
void RigidMapping<BasicMapping>::load(const char *filename)
{
    points.resize(0);

    if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".xs3"))
    {
        Loader loader(this);
        loader.helper::io::MassSpringLoader::load(filename);
    }
    else if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".sph"))
    {
        Loader loader(this);
        loader.helper::io::SphereLoader::load(filename);
    }
    else if (strlen(filename)>0)
    {
        // Default to mesh loader
        helper::io::Mesh* mesh = helper::io::Mesh::Create(filename);
        if (mesh!=NULL)
        {
            points.resize(mesh->getVertices().size());
            for (unsigned int i=0; i<mesh->getVertices().size(); i++)
            {
                Out::DataTypes::set(points[i], mesh->getVertices()[i][0], mesh->getVertices()[i][1], mesh->getVertices()[i][2]);
            }
            delete mesh;
        }
    }
}

template <class BasicMapping>
void RigidMapping<BasicMapping>::init()
{
    if (this->points.empty() && this->toModel!=NULL)
    {
        VecCoord& x = *this->toModel->getX();
        std::cout << "RigidMapping: init "<<x.size()<<" points."<<std::endl;
        points.resize(x.size());
        for (unsigned int i=0; i<x.size(); i++)
            points[i] = x[i];
    }
    this->BasicMapping::init();
}

template <class BasicMapping>
void RigidMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    translation = in[index.getValue()].getCenter();
    in[index.getValue()].writeRotationMatrix(rotation);

    rotatedPoints.resize(points.size());
    out.resize(points.size());
    for(unsigned int i=0; i<points.size(); i++)
    {
        rotatedPoints[i] = rotation*points[i];
        out[i] = rotatedPoints[i];
        out[i] += translation;
    }
}

template <class BasicMapping>
void RigidMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    Deriv v,omega;
    v = in[index.getValue()].getVCenter();
    omega = in[index.getValue()].getVOrientation();
    out.resize(points.size());
    for(unsigned int i=0; i<points.size(); i++)
    {
        // out = J in
        // J = [ I -OM^ ]
        out[i] =  v - cross(rotatedPoints[i],omega);
    }
}

/// Template specialization for 2D rigids
// template<typename real1, typename real2>
// void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::StdRigidTypes<2, real1> >, core::componentmodel::behavior::MechanicalState< defaulttype::StdVectorTypes<defaulttype::Vec<2, real2>, defaulttype::Vec<2, real2>, real2 > > > >::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
// {
//     Deriv v;
//     Real omega;
//     v = in[index.getValue()].getVCenter();
//     omega = (Real)in[index.getValue()].getVOrientation();
//     out.resize(points.size());
//     for(unsigned int i=0;i<points.size();i++)
//     {
//         out[i] =  v + Deriv(-rotatedPoints[i][1],rotatedPoints[i][0])*omega;
//     }
// }

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <class BasicMapping>
void RigidMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    Deriv v,omega;
    for(unsigned int i=0; i<points.size(); i++)
    {
        // out = Jt in
        // Jt = [ I     ]
        //      [ -OM^t ]
        // -OM^t = OM^

        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[index.getValue()].getVCenter() += v;
    out[index.getValue()].getVOrientation() += omega;
}


/// Template specialization for 2D rigids
// template<typename real1, typename real2>
// void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::StdRigidTypes<2, real1> >, core::componentmodel::behavior::MechanicalState< defaulttype::StdVectorTypes<defaulttype::Vec<2, real2>, defaulttype::Vec<2, real2>, real2 > > > >::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
// {
//     Deriv v;
//     Real omega;
//     for(unsigned int i=0;i<points.size();i++)
//     {
//         Deriv f = in[i];
//         v += f;
//         omega += cross(rotatedPoints[i],f);
//     }
//     out[index.getValue()].getVCenter() += v;
//     out[index.getValue()].getVOrientation() += (typename In::Real)omega;
// }

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );


// RigidMapping::applyJT( typename In::VecConst& out, const typename Out::VecConst& in ) //
// this function propagate the constraint through the rigid mapping :
// if one constraint along (vector n) with a value (v) is applied on the childModel (like collision model)
// then this constraint is transformed by (Jt.n) with value (v) for the rigid model
// There is a specificity of this propagateConstraint: we have to find the application point on the childModel
// in order to compute the right constaint on the rigidModel.
template <class BaseMapping>
void RigidMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{

//	printf("\n applyJT(VectConst, VectConst) in RigidMapping");

    out.resize(in.size());

    for(unsigned int i=0; i<in.size(); i++)
    {
        // computation of (Jt.n) //
        // computation of the ApplicationPoint position // Coord is a Vec3
        typename Out::Coord ApplicationPoint;

        // computation of the constaint direction
        typename Out::Deriv n;

        typename Out::Deriv w_n;

        // in[i].size() = num node involved in the constraint
        for (unsigned int j=0; j<in[i].size(); j++)
        {
            int index = in[i][j].index;	// index of the node
            w_n = (Deriv) in[i][j].data;	// weighted value of the constraint direction
            double w = w_n.norm();	// computation of the weight
            // the application point (on the child model) is computed using barycentric values //
            ApplicationPoint += rotatedPoints[index]*w;
            // we add the contribution of each weighted direction
            n += w_n ;
        }

        if (n.norm() < 0.9999 || n.norm() > 1.00001)
            printf("\n WARNING : constraint direction is not normalized !!!");

        // apply Jt.n as a constraint for the center of mass
        // Jt = [ I   ]
        //      [ OM^ ]
        typename Out::Deriv omega_n = cross(ApplicationPoint,n);

        InDeriv direction;

        direction.getVCenter() = n;
        direction.getVOrientation() = omega_n;

        // for rigid model, there's only the center of mass as application point (so only one vector for each constraint)
        out[i].push_back(InSparseDeriv(index.getValue(), direction)); // 0 = index of the center of mass
    }
}

/// Template specialization for 2D rigids

template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );


template <class BasicMapping>
void RigidMapping<BasicMapping>::draw()
{
    if (!getShow(this)) return;
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
