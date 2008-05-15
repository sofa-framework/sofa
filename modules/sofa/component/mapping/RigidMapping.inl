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
#include <string.h>
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
class RigidMapping<BasicMapping>::Loader : public helper::io::MassSpringLoader, public helper::io::SphereLoader
{
public:

    RigidMapping<BasicMapping>* dest;
    Loader(RigidMapping<BasicMapping>* dest) : dest(dest) {}
    virtual void addMass(SReal px, SReal py, SReal pz, SReal, SReal, SReal, SReal, SReal, bool, bool)
    {
        Coord c;
        Out::DataTypes::set(c,px,py,pz);
        dest->points.beginEdit()->push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
    }
    virtual void addSphere(SReal px, SReal py, SReal pz, SReal)
    {
        Coord c;
        Out::DataTypes::set(c,px,py,pz);
        dest->points.beginEdit()->push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
    }
};

template <class BasicMapping>
void RigidMapping<BasicMapping>::load(const char *filename)
{
    points.beginEdit()->resize(0);

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
            points.beginEdit()->resize(mesh->getVertices().size());
            for (unsigned int i=0; i<mesh->getVertices().size(); i++)
            {
                Out::DataTypes::set((*points.beginEdit())[i], mesh->getVertices()[i][0], mesh->getVertices()[i][1], mesh->getVertices()[i][2]);
            }
            delete mesh;
        }
    }
}

template <class BasicMapping>
int RigidMapping<BasicMapping>::addPoint(const Coord& c)
{
    int i = points.getValue().size();
    points.beginEdit()->push_back(c);
    return i;
}

template <class BasicMapping>
int RigidMapping<BasicMapping>::addPoint(const Coord& c, int indexFrom)
{
    int i = points.getValue().size();
    points.beginEdit()->push_back(c);
    if (!repartition.getValue().empty())
    {
        repartition.beginEdit()->push_back(indexFrom);
        repartition.endEdit();
    }
    else if (!i)
    {
        index.setValue(indexFrom);
    }
    else if ((int)index.getValue() != indexFrom)
    {
        sofa::helper::vector<unsigned int>& rep = *repartition.beginEdit();
        rep.clear();
        rep.reserve(i+1);
        rep.insert(rep.end(),index.getValue(),i);
        rep.push_back(indexFrom);
        repartition.endEdit();
    }
    return i;
}

template <class BasicMapping>
void RigidMapping<BasicMapping>::init()
{
    //cerr<<"RigidMapping<BasicMapping>::init begin "<<getName()<<endl;
    if (this->points.getValue().empty() && this->toModel!=NULL)
    {
        VecCoord& x = *this->toModel->getX();
        //std::cout << "RigidMapping: init "<<x.size()<<" points."<<std::endl;
        points.beginEdit()->resize(x.size());
        for (unsigned int i=0; i<x.size(); i++)
            (*points.beginEdit())[i] = x[i];
    }
    //cerr<<"RigidMapping<BasicMapping>::init now doing  BasicMapping::init()"<<getName()<<endl;
    this->BasicMapping::init();
    //cerr<<"RigidMapping<BasicMapping>::init end "<<getName()<<endl;
}
/*
template <class BasicMapping>
void RigidMapping<BasicMapping>::disable()
{

	if (!this->points.getValue().empty() && this->toModel!=NULL)
	{
		VecCoord& x = *this->toModel->getX();
		x.resize(points.getValue().size());
		for (unsigned int i=0;i<points.getValue().size();i++)
			x[i] = points.getValue()[i];
	}
}
*/
template <class BasicMapping>
void RigidMapping<BasicMapping>::clear(int reserve)
{
    this->points.beginEdit()->clear();
    if (reserve) this->points.beginEdit()->reserve(reserve);
    this->repartition.beginEdit()->clear();
    this->repartition.endEdit();
}

template <class BasicMapping>
void RigidMapping<BasicMapping>::setRepartition(unsigned int value)
{
    vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.push_back(value);
    this->repartition.endEdit();
}

template <class BasicMapping>
void RigidMapping<BasicMapping>::setRepartition(sofa::helper::vector<unsigned int> values)
{
    vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.reserve(values.size());
    //repartition.setValue(values);
    sofa::helper::vector<unsigned int>::iterator it = values.begin();
    while (it != values.end())
    {
        rep.push_back(*it);
        it++;
    }
    this->repartition.endEdit();
}

template <class BasicMapping>
void RigidMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    //cerr<<"RigidMapping<BasicMapping>::apply "<<getName()<<endl;
    unsigned int cptOut;
    unsigned int val;
    Coord translation;
    Mat rotation;

    rotatedPoints.resize(points.getValue().size());
    out.resize(points.getValue().size());

    switch (repartition.getValue().size())
    {
    case 0 : //no value specified : simple rigid mapping
        translation = in[index.getValue()].getCenter();
        in[index.getValue()].writeRotationMatrix(rotation);

        for(unsigned int i=0; i<points.getValue().size(); i++)
        {
            rotatedPoints[i] = rotation*points.getValue()[i];
            out[i] = rotatedPoints[i];
            out[i] += translation;
        }
        break;

    case 1 : //one value specified : uniform repartition mapping on the input dofs
        val = repartition.getValue()[0];
        //Out::VecCoord::iterator itOut = out.begin();
        cptOut=0;

        for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            translation = in[ifrom].getCenter();
            in[ifrom].writeRotationMatrix(rotation);

            for(unsigned int ito=0; ito<val; ito++)
            {
                rotatedPoints[cptOut] = rotation* points.getValue()[cptOut];
                out[cptOut] = rotatedPoints[cptOut];
                out[cptOut] += translation;
                cptOut++;
            }
        }
        break;

    default: //n values are specified : heterogen repartition mapping on the input dofs
        if (repartition.getValue().size() != in.size())
        {
            std::cerr<<"Error : mapping dofs repartition is not correct"<<std::endl;
            return;
        }
        cptOut=0;

        for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            translation = in[ifrom].getCenter();
            in[ifrom].writeRotationMatrix(rotation);

            for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
            {
                rotatedPoints[cptOut] = rotation* points.getValue()[cptOut];
                out[cptOut] = rotatedPoints[cptOut];
                out[cptOut] += translation;
                cptOut++;
            }
        }
        break;
    }
}

template <class BasicMapping>
void RigidMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    Deriv v,omega;
    out.resize(points.getValue().size());
    unsigned int cptOut;
    unsigned int val;

    switch (repartition.getValue().size())
    {
    case 0:
        v = in[index.getValue()].getVCenter();
        omega = in[index.getValue()].getVOrientation();
        for(unsigned int i=0; i<points.getValue().size(); i++)
        {
            // out = J in
            // J = [ I -OM^ ]
            out[i] =  v - cross(rotatedPoints[i],omega);
        }
        break;
    case 1:
        val = repartition.getValue()[0];
        cptOut=0;

        for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            v = in[ifrom].getVCenter();
            omega = in[ifrom].getVOrientation();

            for(unsigned int ito=0; ito<val; ito++)
            {
                // out = J in
                // J = [ I -OM^ ]
                out[cptOut] =  v - cross(rotatedPoints[cptOut],omega);
                cptOut++;
            }
        }
        break;
    default:
        if (repartition.getValue().size() != in.size())
        {
            std::cerr<<"Error : mapping dofs repartition is not correct"<<std::endl;
            return;
        }

        cptOut=0;

        for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            v = in[ifrom].getVCenter();
            omega = in[ifrom].getVOrientation();

            for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
            {
                // out = J in
                // J = [ I -OM^ ]
                out[cptOut] =  v - cross(rotatedPoints[cptOut],omega);
                cptOut++;
            }
        }
        break;
    }

}

template <class BasicMapping>
void RigidMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    Deriv v,omega;
    unsigned int val;
    unsigned int cpt;
    switch(repartition.getValue().size())
    {
    case 0 :
        for(unsigned int i=0; i<points.getValue().size(); i++)
        {
            // out = Jt in
            // Jt = [ I     ]
            //      [ -OM^t ]
            // -OM^t = OM^

            Deriv f = in[i];
            //cerr<<"RigidMapping<BasicMapping>::applyJT, f = "<<f<<endl;
            v += f;
            omega += cross(rotatedPoints[i],f);
            //cerr<<"RigidMapping<BasicMapping>::applyJT, new v = "<<v<<endl;
            //cerr<<"RigidMapping<BasicMapping>::applyJT, new omega = "<<omega<<endl;
        }
        out[index.getValue()].getVCenter() += v;
        out[index.getValue()].getVOrientation() += omega;
        break;
    case 1 :
        val = repartition.getValue()[0];
        cpt=0;
        for(unsigned int ito=0; ito<out.size(); ito++)
        {
            v=Deriv();
            omega=Deriv();
            for(unsigned int i=0; i<val; i++)
            {
                Deriv f = in[cpt];
                v += f;
                omega += cross(rotatedPoints[cpt],f);
                cpt++;
            }
            out[ito].getVCenter() += v;
            out[ito].getVOrientation() += omega;
        }
        break;
    default :
        if (repartition.getValue().size() != out.size())
        {
            std::cerr<<"Error : mapping dofs repartition is not correct"<<std::endl;
            return;
        }

        cpt=0;
        for(unsigned int ito=0; ito<out.size(); ito++)
        {
            v=Deriv();
            omega=Deriv();
            for(unsigned int i=0; i<repartition.getValue()[ito]; i++)
            {
                Deriv f = in[cpt];
                v += f;
                omega += cross(rotatedPoints[cpt],f);
                cpt++;
            }
            out[ito].getVCenter() += v;
            out[ito].getVOrientation() += omega;
        }
        break;
    }

}



// RigidMapping::applyJT( typename In::VecConst& out, const typename Out::VecConst& in ) //
// this function propagate the constraint through the rigid mapping :
// if one constraint along (vector n) with a value (v) is applied on the childModel (like collision model)
// then this constraint is transformed by (Jt.n) with value (v) for the rigid model
// There is a specificity of this propagateConstraint: we have to find the application point on the childModel
// in order to compute the right constaint on the rigidModel.
template <class BaseMapping>
void RigidMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{

//	printf("\n applyJT(VectConst, VectConst) in RigidMapping\n");

    int outSize = out.size();
    out.resize(in.size() + outSize); // we can accumulate in "out" constraints from several mappings

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
        out[outSize+i].push_back(InSparseDeriv(index.getValue(), direction)); // 0 = index of the center of mass
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
#ifndef SOFA_FLOAT
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
#endif
#ifndef SOFA_DOUBLE
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
template<>
void RigidMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
#endif
#endif
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



template <class BasicMapping>
void RigidMapping<BasicMapping>::draw()
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
