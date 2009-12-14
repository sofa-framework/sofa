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
#ifndef SOFA_COMPONENT_MAPPING_DEFORMABLEONRIGIDFRAME_INL
#define SOFA_COMPONENT_MAPPING_DEFORMABLEONRIGIDFRAME_INL

#include <sofa/component/mapping/DeformableOnRigidFrameMapping.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/component/container/MultiMeshLoader.h>
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
DeformableOnRigidFrameMapping<BasicMapping>::DeformableOnRigidFrameMapping( In* from, Out* to )
    :Inherit ( from, to )
    , rootModel(NULL)
    , m_rootModelName(initData(&m_rootModelName, std::string(""), "rootModel", "Root position if a rigid root model is specified."))
    , points ( initData ( &points,"initialPoints", "Local Coordinates of the points" ) )
    , index ( initData ( &index, ( unsigned ) 0,"index","input DOF index" ) )
    , fileDeformableOnRigidFrameMapping ( initData ( &fileDeformableOnRigidFrameMapping,"fileDeformableOnRigidFrameMapping","Filename" ) )
    , useX0( initData ( &useX0,false,"useX0","Use x0 instead of local copy of initial positions (to support topo changes)") )
    , indexFromEnd( initData ( &indexFromEnd,false,"indexFromEnd","input DOF index starts from the end of input DOFs vector") )
    , repartition ( initData ( &repartition,"repartition","number of dest dofs per entry dof" ) )
    , globalToLocalCoords ( initData ( &globalToLocalCoords,"globalToLocalCoords","are the output DOFs initially expressed in global coordinates" ) )
{
    this->addAlias(&fileDeformableOnRigidFrameMapping,"filename");
    maskFrom = NULL;
    if (core::componentmodel::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::componentmodel::behavior::BaseMechanicalState *>(from))
        maskFrom = &stateFrom->forceMask;
    maskTo = NULL;
    if (core::componentmodel::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::componentmodel::behavior::BaseMechanicalState *>(to))
        maskTo = &stateTo->forceMask;
}

template <class BasicMapping>
class DeformableOnRigidFrameMapping<BasicMapping>::Loader : public helper::io::MassSpringLoader, public helper::io::SphereLoader
{
public:

    DeformableOnRigidFrameMapping<BasicMapping>* dest;
    Loader(DeformableOnRigidFrameMapping<BasicMapping>* dest) : dest(dest) {}
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
void DeformableOnRigidFrameMapping<BasicMapping>::load(const char *filename)
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
int DeformableOnRigidFrameMapping<BasicMapping>::addPoint(const Coord& c)
{
    int i = points.getValue().size();
    points.beginEdit()->push_back(c);
    return i;
}

template <class BasicMapping>
int DeformableOnRigidFrameMapping<BasicMapping>::addPoint(const Coord& c, int indexFrom)
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
void DeformableOnRigidFrameMapping<BasicMapping>::init()
{
    if ( !fileDeformableOnRigidFrameMapping.getValue().empty() ) this->load ( fileDeformableOnRigidFrameMapping.getFullPath().c_str() );

    if (this->points.getValue().empty() && this->toModel!=NULL && !useX0.getValue())
    {
        VecCoord& x = *this->toModel->getX();
        points.beginEdit()->resize(x.size());
        unsigned int i=0, cpt=0;
        if(globalToLocalCoords.getValue() == true) //test booleen fromWorldCoord
        {
            typename InRoot::VecCoord& xfrom = *this->rootModel->getX();
            switch (repartition.getValue().size())
            {
            case 0 :
                for (i=0; i<x.size(); i++)
                    (*points.beginEdit())[i] = xfrom[0].inverseRotate(x[i]-xfrom[0].getCenter());
                break;
            case 1 :
                for (i=0; i<xfrom.size(); i++)
                    for(unsigned int j=0; j<repartition.getValue()[0]; j++,cpt++)
                        (*points.beginEdit())[cpt] = xfrom[i].inverseRotate(x[cpt]-xfrom[i].getCenter());
                break;
            default :
                for (i=0; i<xfrom.size(); i++)
                    for(unsigned int j=0; j<repartition.getValue()[i]; j++,cpt++)
                        (*points.beginEdit())[cpt] = xfrom[i].inverseRotate(x[cpt]-xfrom[i].getCenter());
                break;
            }
        }
        else
        {
            for (i=0; i<x.size(); i++)
                (*points.beginEdit())[i] = x[i];
        }
    }

    this->BasicMapping::init();

    sofa::component::container::MultiMeshLoader * loader;
    this->getContext()->get(loader);
    if (loader)
    {
        sofa::helper::vector<unsigned int>& rep = *repartition.beginEdit();
        unsigned int cpt=0;
        typename InRoot::VecCoord& xfrom = *this->rootModel->getX();
        VecCoord& xto = *this->toModel->getX();
        for (unsigned int i=0 ; i<loader->getNbMeshs() ; i++)
        {
            rep.push_back(loader->getNbPoints(i));
            if(globalToLocalCoords.getValue() == true)
            {
                for (unsigned int j=0 ; j<loader->getNbPoints(i) ; j++, cpt++)
                {
                    (*points.beginEdit())[cpt] = xfrom[i].inverseRotate(xto[cpt]-xfrom[i].getCenter());
                }
            }
        }
        repartition.endEdit();
    }

    if (!m_rootModelName.getValue().empty())
    {
        std::vector< std::string > tokens(0);
        std::string path = m_rootModelName.getValue();

        this->fromModel->getContext()->get(rootModel , path);
    }
    else
        this->fromModel->getContext()->get(rootModel, core::objectmodel::BaseContext::SearchUp);

    if (rootModel)
        std::cout << "Root Model found : Name = " << rootModel->getName() << sendl;
    else
        std::cerr << " NO ROOT MODEL FOUND"<<sendl;
}
/*
template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::disable()
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
void DeformableOnRigidFrameMapping<BasicMapping>::clear(int reserve)
{
    this->points.beginEdit()->clear();
    if (reserve) this->points.beginEdit()->reserve(reserve);
    this->repartition.beginEdit()->clear();
    this->repartition.endEdit();
}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::setRepartition(unsigned int value)
{
    vector<unsigned int>& rep = *this->repartition.beginEdit();
    rep.clear();
    rep.push_back(value);
    this->repartition.endEdit();
}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::setRepartition(sofa::helper::vector<unsigned int> values)
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
const typename DeformableOnRigidFrameMapping<BasicMapping>::VecCoord & DeformableOnRigidFrameMapping<BasicMapping>::getPoints()
{
    if(useX0.getValue())
    {
        const VecCoord* v = M_getX0(this->toModel);
        if (v) return *v;
        else serr << "DeformableOnRigidFrameMapping: ERROR useX0 can only be used in MechanicalMappings." << sendl;
    }
    return points.getValue();
}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::apply( typename Out::VecCoord& /* out */ , const typename In::VecCoord& /* in */ , const typename InRoot::VecCoord* /* inroot  */ )
{

}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::applyJ( typename Out::VecDeriv& /* out*/ , const typename In::VecDeriv& /* in */ , const typename InRoot::VecDeriv* /* inroot */ )
{

}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::applyJT( typename In::VecDeriv& /* out */, const typename Out::VecDeriv& /* in */, typename InRoot::VecDeriv* /* outroot */)
{

}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::applyJT( typename In::VecConst& /* out */, const typename Out::VecConst& /* in */, typename InRoot::VecConst* /* outroot */)
{

}


/// Template specialization for 2D rigids
// template<typename real1, typename real2>
// void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::StdRigidTypes<2, real1> >, core::componentmodel::behavior::MechanicalState< defaulttype::StdVectorTypes<defaulttype::Vec<2, real2>, defaulttype::Vec<2, real2>, real2 > > > >::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
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
//#ifndef SOFA_FLOAT
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
//#endif
//#ifndef SOFA_DOUBLE
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
//#endif
//
//#ifndef SOFA_FLOAT
//#ifndef SOFA_DOUBLE
//template<>
//void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::componentmodel::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
//#endif
//#endif
/// Template specialization for 2D rigids
// template<typename real1, typename real2>
// void DeformableOnRigidFrameMapping< core::componentmodel::behavior::MechanicalMapping< core::componentmodel::behavior::MechanicalState< defaulttype::StdRigidTypes<2, real1> >, core::componentmodel::behavior::MechanicalState< defaulttype::StdVectorTypes<defaulttype::Vec<2, real2>, defaulttype::Vec<2, real2>, real2 > > > >::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
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
void DeformableOnRigidFrameMapping<BasicMapping>::draw()
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
