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
#ifndef SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_INL
#define SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_INL

#include <sofa/component/mapping/RigidRigidMapping.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/helper/gl/Axis.h>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;



namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

template <class BasicMapping>
class RigidRigidMapping<BasicMapping>::Loader : public helper::io::MassSpringLoader, public helper::io::SphereLoader
{
public:
    RigidRigidMapping<BasicMapping>* dest;
    Loader(RigidRigidMapping<BasicMapping>* dest) : dest(dest) {}
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
void RigidRigidMapping<BasicMapping>::load(const char *filename)
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
void RigidRigidMapping<BasicMapping>::init()
{
    if (this->points.empty() && this->toModel!=NULL)
    {
        VecCoord& x = *this->toModel->getX();
        points.resize(x.size());
        for (unsigned int i=0; i<x.size(); i++)
            points[i] = x[i];
    }
    this->BasicMapping::init();
}

template <class BasicMapping>
void RigidRigidMapping<BasicMapping>::clear()
{
    this->points.clear();
}

template <class BasicMapping>
void RigidRigidMapping<BasicMapping>::setRepartition(unsigned int value)
{
    vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.push_back(value);
    this->repartition.endEdit();
}

template <class BasicMapping>
void RigidRigidMapping<BasicMapping>::setRepartition(std::vector<unsigned int> values)
{
    vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.reserve(values.size());
    //repartition.setValue(values);
    std::vector<unsigned int>::iterator it = values.begin();
    while (it != values.end())
    {
        rep.push_back(*it);
        it++;
    }
    this->repartition.endEdit();
}

template <class BasicMapping>
void RigidRigidMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    unsigned int cptOut;
    unsigned int val;

    out.resize(points.size());
    pointsR0.resize(points.size());

    switch (repartition.getValue().size())
    {
    case 0 : //no value specified : simple rigid mapping
        in[index.getValue()].writeRotationMatrix(rotation);
        for(unsigned int i=0; i<points.size(); i++)
        {
            pointsR0[i].getCenter() = rotation*(points[i]).getCenter();
            out[i] = in[index.getValue()].mult(points[i]);
        }
        break;

    case 1 : //one value specified : uniform repartition.getValue() mapping on the input dofs
        val = repartition.getValue()[0];
        cptOut=0;

        for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            in[ifrom].writeRotationMatrix(rotation);
            for(unsigned int ito=0; ito<val; ito++)
            {
                pointsR0[cptOut].getCenter() = rotation*(points[cptOut]).getCenter();
                out[cptOut] = in[ifrom].mult(points[cptOut]);
                cptOut++;
            }
        }
        break;

    default: //n values are specified : heterogen repartition.getValue() mapping on the input dofs
        if (repartition.getValue().size() != in.size())
        {
            std::cerr<<"Error : mapping dofs repartition.getValue() is not correct"<<std::endl;
            return;
        }
        cptOut=0;

        for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            in[ifrom].writeRotationMatrix(rotation);
            for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
            {
                pointsR0[cptOut].getCenter() = rotation*(points[cptOut]).getCenter();
                out[cptOut] = in[ifrom].mult(points[cptOut]);
                cptOut++;
            }
        }
        break;
    }
}

template <class BasicMapping>
void RigidRigidMapping<BasicMapping>::applyJ( typename Out::VecDeriv& childForces, const typename In::VecDeriv& parentForces )
{
    Vec v,omega;
    childForces.resize(points.size());
    unsigned int cptchildForces;
    unsigned int val;

    switch (repartition.getValue().size())
    {
    case 0:
        v = parentForces[index.getValue()].getVCenter();
        omega = parentForces[index.getValue()].getVOrientation();
        for(unsigned int i=0; i<points.size(); i++)
        {
            childForces[i].getVCenter() =  v + cross(omega,pointsR0[i].getCenter());
            childForces[i].getVOrientation() = omega;
        }
        break;

    case 1:
        val = repartition.getValue()[0];
        cptchildForces=0;
        for (unsigned int ifrom=0 ; ifrom<parentForces.size() ; ifrom++)
        {
            v = parentForces[ifrom].getVCenter();
            omega = parentForces[ifrom].getVOrientation();

            for(unsigned int ito=0; ito<val; ito++)
            {
                childForces[cptchildForces].getVCenter() =  v + cross(omega,(pointsR0[cptchildForces]).getCenter());
                childForces[cptchildForces].getVOrientation() = omega;
                cptchildForces++;
            }
        }
        break;

    default:
        if (repartition.getValue().size() != parentForces.size())
        {
            std::cerr<<"Error : mapping dofs repartition.getValue() is not correct"<<std::endl;
            return;
        }
        cptchildForces=0;
        for (unsigned int ifrom=0 ; ifrom<parentForces.size() ; ifrom++)
        {
            v = parentForces[ifrom].getVCenter();
            omega = parentForces[ifrom].getVOrientation();

            for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
            {
                childForces[cptchildForces].getVCenter() =  v + cross(omega,(pointsR0[cptchildForces]).getCenter());
                childForces[cptchildForces].getVOrientation() = omega;
                cptchildForces++;
            }
        }
        break;
    }

}


template <class BasicMapping>
void RigidRigidMapping<BasicMapping>::applyJT( typename In::VecDeriv& parentForces, const typename Out::VecDeriv& childForces )
{
    Vec v,omega;
    unsigned int val;
    unsigned int cpt;
    switch(repartition.getValue().size())
    {
    case 0 :
        for(unsigned int i=0; i<points.size(); i++)
        {
            // out = Jt in
            // Jt = [ I     ]
            //      [ -OM^t ]
            // -OM^t = OM^

            Vec f = childForces[i].getVCenter();
            v += f;
            omega += childForces[i].getVOrientation() + cross(f,-pointsR0[i].getCenter());
        }
        parentForces[index.getValue()].getVCenter() += v;
        parentForces[index.getValue()].getVOrientation() += omega;
        break;
    case 1 :
        val = repartition.getValue()[0];
        cpt=0;
        for(unsigned int ito=0; ito<parentForces.size(); ito++)
        {
            v=Vec();
            omega=Vec();
            for(unsigned int i=0; i<val; i++)
            {
                Vec f = childForces[cpt].getVCenter();
                v += f;
                omega += childForces[cpt].getVOrientation() + cross(f,-pointsR0[cpt].getCenter());
                cpt++;
            }
            parentForces[ito].getVCenter() += v;
            parentForces[ito].getVOrientation() += omega;
        }
        break;
    default :
        if (repartition.getValue().size() != parentForces.size())
        {
            std::cerr<<"Error : mapping dofs repartition.getValue() is not correct"<<std::endl;
            return;
        }
        cpt=0;
        for(unsigned int ito=0; ito<parentForces.size(); ito++)
        {
            v=Vec();
            omega=Vec();
            for(unsigned int i=0; i<repartition.getValue()[ito]; i++)
            {
                Vec f = childForces[cpt].getVCenter();
                v += f;
                omega += childForces[cpt].getVOrientation() + cross(f,-pointsR0[cpt].getCenter());
                cpt++;
            }
            parentForces[ito].getVCenter() += v;
            parentForces[ito].getVOrientation() += omega;

        }
        break;
    }

}


template <class BasicMapping>
void RigidRigidMapping<BasicMapping>::draw()
{
    if (!getShow(this)) return;
    const typename Out::VecCoord& x = *this->toModel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::Axis::draw(x[i].getCenter(), x[i].getOrientation(), 0.7);
    }
    glEnd();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
