/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/core/Mapping.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/simulation/common/Simulation.h>
#include <string.h>
#include <iostream>






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
    if ( !fileRigidMapping.getValue().empty() ) this->load ( fileRigidMapping.getFullPath().c_str() );
    //serr<<"RigidMapping<BasicMapping>::init begin "<<getName()<<sendl;
    if (this->points.getValue().empty() && this->toModel!=NULL && !useX0.getValue())
    {
        VecCoord& x = *this->toModel->getX();
        //sout << "RigidMapping: init "<<x.size()<<" points."<<sendl;
        points.beginEdit()->resize(x.size());
        for (unsigned int i=0; i<x.size(); i++)
            (*points.beginEdit())[i] = x[i];
    }
    //serr<<"RigidMapping<BasicMapping>::init now doing  BasicMapping::init()"<<getName()<<sendl;
    this->BasicMapping::init();
    //serr<<"RigidMapping<BasicMapping>::init end "<<getName()<<sendl;
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

template<class DataTypes>
const typename DataTypes::VecCoord* M_getX0(core::componentmodel::behavior::MechanicalState<DataTypes>* model)
{
    return model->getX0();
}

template<class DataTypes>
const typename DataTypes::VecCoord* M_getX0(core::componentmodel::behavior::MappedModel<DataTypes>* /*model*/)
{
    return NULL;
}

template <class BasicMapping>
const typename RigidMapping<BasicMapping>::VecCoord & RigidMapping<BasicMapping>::getPoints()
{
    if(useX0.getValue())
    {
        const VecCoord* v = M_getX0(this->toModel);
        if (v) return *v;
        else serr << "RigidMapping: ERROR useX0 can only be used in MechanicalMappings." << sendl;
    }
    return points.getValue();
}

template <class BasicMapping>
void RigidMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    //serr<<"RigidMapping<BasicMapping>::apply "<<getName()<<sendl;
    unsigned int cptOut;
    unsigned int val;
    Coord translation;
    Mat rotation;

    const VecCoord& pts = this->getPoints();

    rotatedPoints.resize(pts.size());
    out.resize(pts.size());

    switch (repartition.getValue().size())
    {
    case 0 : //no value specified : simple rigid mapping

        if (indexFromEnd.getValue())
        {
            translation = in[in.size() - 1 - index.getValue()].getCenter();
            in[in.size() - 1 - index.getValue()].writeRotationMatrix(rotation);
        }
        else
        {
            translation = in[index.getValue()].getCenter();
            in[index.getValue()].writeRotationMatrix(rotation);
        }

        for(unsigned int i=0; i<pts.size(); i++)
        {
            rotatedPoints[i] = rotation*pts[i];
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
                rotatedPoints[cptOut] = rotation* pts[cptOut];
                out[cptOut] = rotatedPoints[cptOut];
                out[cptOut] += translation;
                cptOut++;
            }
        }
        break;

    default: //n values are specified : heterogen repartition mapping on the input dofs
        if (repartition.getValue().size() != in.size())
        {
            serr<<"Error : mapping dofs repartition is not correct"<<sendl;
            return;
        }
        cptOut=0;

        for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
        {
            translation = in[ifrom].getCenter();
            in[ifrom].writeRotationMatrix(rotation);

            for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
            {
                rotatedPoints[cptOut] = rotation* pts[cptOut];
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
    const VecCoord& pts = this->getPoints();
    out.resize(pts.size());
    unsigned int cptOut;
    unsigned int val;


    if ( !maskTo || !(maskTo->isInUse()) )
    {
        switch (repartition.getValue().size())
        {
        case 0:
            if (indexFromEnd.getValue())
            {
                v = in[in.size() - 1 - index.getValue()].getVCenter();
                omega = in[in.size() - 1 - index.getValue()].getVOrientation();
            }
            else
            {
                v = in[index.getValue()].getVCenter();
                omega = in[index.getValue()].getVOrientation();
            }

            for(unsigned int i=0; i<pts.size(); i++)
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
                serr<<"Error : mapping dofs repartition is not correct"<<sendl;
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
    else
    {
        switch (repartition.getValue().size())
        {
        case 0:
        {
            if (indexFromEnd.getValue())
            {
                v = in[in.size() - 1 - index.getValue()].getVCenter();
                omega = in[in.size() - 1 - index.getValue()].getVOrientation();
            }
            else
            {
                v = in[index.getValue()].getVCenter();
                omega = in[index.getValue()].getVOrientation();
            }

            typedef core::componentmodel::behavior::BaseMechanicalState::ParticleMask ParticleMask;
            const ParticleMask::InternalStorage &indices=maskTo->getEntries();

            ParticleMask::InternalStorage::const_iterator it;
            for (it=indices.begin(); it!=indices.end(); it++)
            {
                const int i=(int)(*it);

                // out = J in
                // J = [ I -OM^ ]
                out[i] =  v - cross(rotatedPoints[i],omega);
            }
            break;
        }
        case 1:
        {
            val = repartition.getValue()[0];
            cptOut=0;

            typedef core::componentmodel::behavior::BaseMechanicalState::ParticleMask ParticleMask;
            const ParticleMask::InternalStorage &indices=maskTo->getEntries();

            for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
            {
                v = in[ifrom].getVCenter();
                omega = in[ifrom].getVOrientation();

                for(unsigned int ito=0; ito<val; ito++)
                {
                    // out = J in
                    // J = [ I -OM^ ]
                    if (indices.find( cptOut) != indices.end())
                    {
                        out[cptOut] =  v - cross(rotatedPoints[cptOut],omega);
                    }
                    cptOut++;
                }
            }
            break;
        }
        default:
        {
            if (repartition.getValue().size() != in.size())
            {
                serr<<"Error : mapping dofs repartition is not correct"<<sendl;
                return;
            }

            cptOut=0;

            typedef core::componentmodel::behavior::BaseMechanicalState::ParticleMask ParticleMask;
            const ParticleMask::InternalStorage &indices=maskTo->getEntries();

            for (unsigned int ifrom=0 ; ifrom<in.size() ; ifrom++)
            {
                v = in[ifrom].getVCenter();
                omega = in[ifrom].getVOrientation();

                for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
                {
                    // out = J in
                    // J = [ I -OM^ ]
                    if (indices.find( cptOut) != indices.end())
                    {
                        out[cptOut] =  v - cross(rotatedPoints[cptOut],omega);
                    }
                    cptOut++;
                }
            }
            break;
        }
        }
    }

}

template <class BasicMapping>
void RigidMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    Deriv v,omega;
    unsigned int val;
    unsigned int cpt;
    const VecCoord& pts = this->getPoints();


    if (  !maskTo || !(maskTo->isInUse()) )
    {
        switch(repartition.getValue().size())
        {
        case 0 :
            for(unsigned int i=0; i<pts.size(); i++)
            {
                // out = Jt in
                // Jt = [ I     ]
                //      [ -OM^t ]
                // -OM^t = OM^

                Deriv f = in[i];
                //serr<<"RigidMapping<BasicMapping>::applyJT, f = "<<f<<sendl;
                v += f;
                omega += cross(rotatedPoints[i],f);
                //serr<<"RigidMapping<BasicMapping>::applyJT, new v = "<<v<<sendl;
                //serr<<"RigidMapping<BasicMapping>::applyJT, new omega = "<<omega<<sendl;
            }

            if (indexFromEnd.getValue())
            {
                out[out.size() - 1 - index.getValue()].getVCenter() += v;
                out[out.size() - 1 - index.getValue()].getVOrientation() += omega;
            }
            else
            {
                out[index.getValue()].getVCenter() += v;
                out[index.getValue()].getVOrientation() += omega;
            }

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
                serr<<"Error : mapping dofs repartition is not correct"<<sendl;
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
    else
    {
        switch(repartition.getValue().size())
        {
        case 0 :
        {
            typedef core::componentmodel::behavior::BaseMechanicalState::ParticleMask ParticleMask;
            const ParticleMask::InternalStorage &indices=maskTo->getEntries();

            ParticleMask::InternalStorage::const_iterator it;
            for (it=indices.begin(); it!=indices.end(); it++)
            {
                const int i=(int)(*it);
                // out = Jt in
                // Jt = [ I     ]
                //      [ -OM^t ]
                // -OM^t = OM^

                Deriv f = in[i];
                //serr<<"RigidMapping<BasicMapping>::applyJT, f = "<<f<<sendl;
                v += f;
                omega += cross(rotatedPoints[i],f);
                //serr<<"RigidMapping<BasicMapping>::applyJT, new v = "<<v<<sendl;
                //serr<<"RigidMapping<BasicMapping>::applyJT, new omega = "<<omega<<sendl;
            }

            if (indexFromEnd.getValue())
            {
                out[out.size() - 1 - index.getValue()].getVCenter() += v;
                out[out.size() - 1 - index.getValue()].getVOrientation() += omega;
            }
            else
            {
                out[index.getValue()].getVCenter() += v;
                out[index.getValue()].getVOrientation() += omega;
            }

            break;
        }
        case 1 :
        {
            val = repartition.getValue()[0];
            cpt=0;
            typedef core::componentmodel::behavior::BaseMechanicalState::ParticleMask ParticleMask;
            const ParticleMask::InternalStorage &indices=maskTo->getEntries();

            for(unsigned int ito=0; ito<out.size(); ito++)
            {
                v=Deriv();
                omega=Deriv();
                for(unsigned int i=0; i<val; i++)
                {
                    if (indices.find(cpt) != indices.end())
                    {
                        Deriv f = in[cpt];
                        v += f;
                        omega += cross(rotatedPoints[cpt],f);
                    }
                    cpt++;
                }
                out[ito].getVCenter() += v;
                out[ito].getVOrientation() += omega;
            }
            break;
        }
        default :
        {
            if (repartition.getValue().size() != out.size())
            {
                serr<<"Error : mapping dofs repartition is not correct"<<sendl;
                return;
            }

            cpt=0;
            typedef core::componentmodel::behavior::BaseMechanicalState::ParticleMask ParticleMask;
            const ParticleMask::InternalStorage &indices=maskTo->getEntries();

            for(unsigned int ito=0; ito<out.size(); ito++)
            {
                v=Deriv();
                omega=Deriv();
                for(unsigned int i=0; i<repartition.getValue()[ito]; i++)
                {
                    if (indices.find(cpt) != indices.end())
                    {
                        Deriv f = in[cpt];
                        v += f;
                        omega += cross(rotatedPoints[cpt],f);
                    }
                    cpt++;
                }
                out[ito].getVCenter() += v;
                out[ito].getVOrientation() += omega;
            }
        }
        break;
        }
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

        OutConstraintIterator itOut;
        for (itOut=in[i].getData().begin(); itOut!=in[i].getData().end(); itOut++)
        {
            unsigned int indexIn = itOut->first;// index of the node
            Deriv data=(Deriv) itOut->second;

            w_n = (Deriv) data;	// weighted value of the constraint direction
            double w = w_n.norm();	// computation of the weight
            // the application point (on the child model) is computed using barycentric values //
            ApplicationPoint += rotatedPoints[indexIn]*w;
            // we add the contribution of each weighted direction
            n += w_n ;
        }

//   		if (n.norm() < 0.9999 || n.norm() > 1.00001)
//  			printf("\n WARNING : constraint direction is not normalized !!!\n");

        // apply Jt.n as a constraint for the center of mass
        // Jt = [ I   ]
        //      [ OM^ ]
//                 typename Out::Deriv _n=n; _n.normalize();
        typename Out::Deriv omega_n = cross(ApplicationPoint,n);

        InDeriv direction;

        direction.getVCenter() = n;
        direction.getVOrientation() = omega_n;

        switch(repartition.getValue().size())
        {
        case 0 :
            // for rigid model, there's only the center of mass as application point (so only one vector for each constraint)
            if (indexFromEnd.getValue())
            {
                out[outSize+i].insert(out.size() - 1 - index.getValue(), direction); // 0 = index of the center of mass
            }
            else
            {
                out[outSize+i].insert(index.getValue(), direction); // 0 = index of the center of mass
            }
            break;
        case 1:
            serr << "applyJT with repartition NOT SUPPORTED YET" << sendl;
            break;
        default:
            serr << "applyJT with repartition NOT SUPPORTED YET" << sendl;
            break;
        }
    }
}

template <class BaseMapping>
void RigidMapping<BaseMapping>::applyJT( core::componentmodel::behavior::BaseMechanicalState::ParticleMask& out, const core::componentmodel::behavior::BaseMechanicalState::ParticleMask& in )
{

    typedef core::componentmodel::behavior::BaseMechanicalState::ParticleMask ParticleMask;
    const ParticleMask::InternalStorage &indices=in.getEntries();

    ParticleMask::InternalStorage::const_iterator it;

    switch(repartition.getValue().size())
    {
    case 0 :
    {
        if (indices.size()) out.insertEntry(index.getValue());
        return;
        break;
    }
    case 1 :
    {
        const unsigned int val = repartition.getValue()[0];
        for (it=indices.begin(); it!=indices.end(); it++)
        {
            const unsigned int idx=(*it);
            if (idx < val)
            {
                out.insertEntry(index.getValue());
                return;
            }
        }
        break;
    }
    default:
    {
        unsigned cpt=0;
        unsigned idxRigid=0;
        unsigned int numSubDiv = repartition.getValue().size();
        for (it=indices.begin(); it!=indices.end(); it++)
        {
            const unsigned int index=(*it);
            while (index > cpt && idxRigid < numSubDiv)
            {
                cpt += repartition.getValue()[idxRigid];
                idxRigid++;
            }

            out.insertEntry(idxRigid);
        }
        break;
    }
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
    std::vector< Vector3 > points;
    Vector3 point;
    unsigned int sizePoints= (Coord::static_size <=3)?Coord::static_size:3;

    const typename Out::VecCoord& x = *this->toModel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        for (unsigned int s=0; s<sizePoints; ++s) point[s] = x[i][s];
        points.push_back(point);
    }
    simulation::getSimulation()->DrawUtility.drawPoints(points, 7, Vec<4,float>(1,1,0,1));
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
