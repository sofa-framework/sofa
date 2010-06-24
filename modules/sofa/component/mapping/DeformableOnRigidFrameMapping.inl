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
#include <sofa/core/behavior/MechanicalMapping.inl>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/behavior/MappedModel.h>
#include <sofa/simulation/common/Simulation.h>
#include <string.h>
#include <iostream>


/*!
 *   This mapping is derived from the RigidMapping. The difference is :
 *   In the RigidMapping, the rigid is considered like a perfect rigid (non-deformable)
 *   In this one, the rigid allow a low deformation of the rigid.
 *
 *   Principale difference with the RigidMapping is in the fonctions apply, applyJ and applyJT
 */

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
//, points ( initData ( &points,"initialPoints", "Local Coordinates of the points" ) )
    , index ( initData ( &index, ( unsigned ) 0,"index","input DOF index" ) )
    , fileDeformableOnRigidFrameMapping ( initData ( &fileDeformableOnRigidFrameMapping,"fileDeformableOnRigidFrameMapping","Filename" ) )
    , useX0( initData ( &useX0,false,"useX0","Use x0 instead of local copy of initial positions (to support topo changes)") )
    , indexFromEnd( initData ( &indexFromEnd,false,"indexFromEnd","input DOF index starts from the end of input DOFs vector") )
    , repartition ( initData ( &repartition,"repartition","number of dest dofs per entry dof" ) )
    , globalToLocalCoords ( initData ( &globalToLocalCoords,"globalToLocalCoords","are the output DOFs initially expressed in global coordinates" ) )
{
    this->addAlias(&fileDeformableOnRigidFrameMapping,"filename");
    maskFrom = NULL;
    if (core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *>(from))
        maskFrom = &stateFrom->forceMask;
    maskTo = NULL;
    if (core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *>(to))
        maskTo = &stateTo->forceMask;
}

template <class BasicMapping>
class DeformableOnRigidFrameMapping<BasicMapping>::Loader : public helper::io::MassSpringLoader, public helper::io::SphereLoader
{
public:

    DeformableOnRigidFrameMapping<BasicMapping>* dest;
    Loader(DeformableOnRigidFrameMapping<BasicMapping>* dest) : dest(dest) {}

};

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::load(const char * /*filename*/)
{

}

template <class BasicMapping>
int DeformableOnRigidFrameMapping<BasicMapping>::addPoint(const Coord& /*c*/)
{
    //int i = points.getValue().size();
    //points.beginEdit()->push_back(c);
    std::cout<<"addPoint should be supress"<<std::endl;
    return 0;
}

template <class BasicMapping>
int DeformableOnRigidFrameMapping<BasicMapping>::addPoint(const Coord& /*c*/, int /*indexFrom*/)
{
    std::cout<<"addPoint should be supress"<<std::endl;
    return 0;
}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::init()
{


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

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::clear(int /*reserve*/)
{

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
void DeformableOnRigidFrameMapping<BasicMapping>::setRepartition(sofa::helper::vector<unsigned int> /*values*/)
{

}

template<class DataTypes>
const typename DataTypes::VecCoord* M_getX0(core::behavior::MechanicalState<DataTypes>* model)
{
    return model->getX0();
}

template<class DataTypes>
const typename DataTypes::VecCoord* M_getX0(core::behavior::MappedModel<DataTypes>* /*model*/)
{
    return NULL;
}

/*template <class BasicMapping>
const typename DeformableOnRigidFrameMapping<BasicMapping>::VecCoord & DeformableOnRigidFrameMapping<BasicMapping>::getPoints()
{

        return VecCoord();
}
*/
template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::apply( typename Out::VecCoord& out  , const typename In::VecCoord & inDeformed  , const typename InRoot::VecCoord * inRigid  )
{
//Fin the rigid center[s] and its displacement
//Apply the displacement to all the located points

    //std::cout<<"+++++++++ apply is called"<<std::endl;

    if (rootModel)
    {
        unsigned int cptOut;
        unsigned int val;
        Coord translation;
        Mat rotation;

        rotatedPoints.resize(inDeformed.size());
        out.resize(inDeformed.size());
        switch (repartition.getValue().size())
        {
        case 0 :
            if (indexFromEnd.getValue())
            {
                translation = (*inRigid)[(*inRigid).size() - 1 - index.getValue()].getCenter();
                (*inRigid)[(*inRigid).size() - 1 - index.getValue()].writeRotationMatrix(rotation);
                rootX = (*inRigid)[(*inRigid).size() - 1 - index.getValue()];
            }
            else
            {
                translation = (*inRigid)[index.getValue()].getCenter();
                (*inRigid)[index.getValue()].writeRotationMatrix(rotation);
                rootX = (*inRigid)[index.getValue()];
            }

            for(unsigned int i=0; i<inDeformed.size(); i++)
            {
                rotatedPoints[i] = rotation*inDeformed[i];
                out[i] = rotatedPoints[i];
                out[i] += translation;
            }
            break;

        case 1 ://one value specified : uniform repartition mapping on the input dofs
            val = repartition.getValue()[0];
            cptOut=0;
            for (unsigned int ifrom=0 ; ifrom<(*inRigid).size() ; ifrom++)
            {
                translation = (*inRigid)[ifrom].getCenter();
                (*inRigid)[ifrom].writeRotationMatrix(rotation);
                rootX = (*inRigid)[ifrom];

                for(unsigned int ito=0; ito<val; ito++)
                {
                    rotatedPoints[cptOut] = rotation*inDeformed[cptOut];
                    out[cptOut] = rotatedPoints[cptOut];
                    out[cptOut] += translation;
                    cptOut++;
                }
            }
            break;

        default :
            if (repartition.getValue().size() != (*inRigid).size())
            {
                serr<<"Error : mapping dofs repartition is not correct"<<sendl;
                return;
            }
            cptOut=0;

            for (unsigned int ifrom=0 ; ifrom<(*inRigid).size() ; ifrom++)
            {
                translation = (*inRigid)[ifrom].getCenter();
                (*inRigid)[ifrom].writeRotationMatrix(rotation);
                rootX = (*inRigid)[ifrom];

                for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++)
                {
                    rotatedPoints[cptOut] = rotation*inDeformed[cptOut];
                    out[cptOut] = rotatedPoints[cptOut];
                    out[cptOut] += translation;
                    cptOut++;
                }
            }
            break;
        }
    }
    else  // no rootModel found => mapping is identity !
    {
        rootX = InRoot::Coord();
        out.resize(inDeformed.size()); rotatedPoints.resize(inDeformed.size());
        for(unsigned int i=0; i<inDeformed.size(); i++)
        {
            rotatedPoints[i] = inDeformed[i];
            out[i] = rotatedPoints[i];
        }
    }
}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::applyJ( typename Out::VecDeriv&  out , const typename In::VecDeriv& inDeformed , const typename InRoot::VecDeriv* inRigid)
{
    if (rootModel)
    {
        Deriv v,omega;//Vec3d
        out.resize(inDeformed.size());
        //unsigned int cptOut;
        //unsigned int val;


        //switch (repartition.getValue().size())
        //  {
        //  case 0:
        if (indexFromEnd.getValue())
        {
            v = (*inRigid)[(*inRigid).size() - 1 - index.getValue()].getVCenter();
            omega = (*inRigid)[(*inRigid).size() - 1 - index.getValue()].getVOrientation();
        }
        else
        {
            v = (*inRigid)[index.getValue()].getVCenter();
            omega = (*inRigid)[index.getValue()].getVOrientation();
        }


        for(unsigned int i=0; i<inDeformed.size(); i++)
        {
            out[i] = cross(omega,rotatedPoints[i]);
            out[i] += rootX.getOrientation().rotate(inDeformed[i]); //velocity on the local system : (Vrigid + Vdeform)
            out[i]+= v; //center velocity
        }
        //         break;
        /* case 1://one value specified : uniform repartition mapping on the input dofs
                 val = repartition.getValue()[0];
                 cptOut=0;
                 for (unsigned int ifrom=0 ; ifrom<(*inRigid).size() ; ifrom++){
                         v = (*inRigid)[ifrom].getVCenter();
                         omega = (*inRigid)[ifrom].getVOrientation();

                         for(unsigned int ito=0; ito<val; ito++){
                                 out[cptOut] = -cross(rotatedPoints[ito],omega)+ rootX.getOrientation().rotate(inDeformed[ito]);
                                 out[cptOut] += v;
                                 cptOut++;
                         }
                 }
                 break;
         default:
                 if (repartition.getValue().size() != (*inRigid).size()){
                         serr<<"Error : mapping dofs repartition is not correct"<<sendl;
                         return;
                 }
                 cptOut=0;
                 for (unsigned int ifrom=0 ; ifrom<(*inRigid).size() ; ifrom++){
                         v = (*inRigid)[ifrom].getVCenter();
                         omega = (*inRigid)[ifrom].getVOrientation();

                         for(unsigned int ito=0; ito<repartition.getValue()[ifrom]; ito++){
                                 out[cptOut] = -cross(rotatedPoints[cptOut],omega) + rootX.getOrientation().rotate(inDeformed[cptOut]);
                                 out[cptOut] += v;
                                 cptOut++;
                         }
                 }
                 break;
         }
         */
    }



    else // no root model!
    {
        serr<<"NO ROOT MODEL"<<sendl;
        out.resize(inDeformed.size());
        for(unsigned int i=0; i<inDeformed.size(); i++)
        {
            out[i] = inDeformed[i];
        }
    }
}
template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in, typename InRoot::VecDeriv* outRoot)
{

    if (rootModel)
    {
        Deriv v,omega;
//	unsigned int val;
//	unsigned int cpt;
//	const VecCoord& pts = this->getPoints();
//        out.resize(in.size());


        // switch (repartition.getValue().size())
        //{
        //    case 0:
        //   std::cout<<"case 0"<<std::endl;
        //std::cout<<" in.size() = "<<in.size()<<"  rotatedPoint.size()" <<rotatedPoints.size()<<std::endl;


        if (in.size() != rotatedPoints.size())
        {
            bool log = this->f_printLog.getValue();
            //std::cout<<"+++++++++++ LOG +++++++++ "<<log<<std::endl;
            //this->f_printLog.setValue(true);
            //serr<<"Warning: applyJT was called before any apply"<<sendl;
            this->propagateX();
            this->f_printLog.setValue(log);
        }

        for(unsigned int i=0; i<in.size(); i++)
        {
            Deriv f = in[i];
            v += f;
            omega += cross(rotatedPoints[i],f);
        }

        if (indexFromEnd.getValue())
        {

            (*outRoot)[(*outRoot).size() - 1 - index.getValue()].getVCenter() += v;
            (*outRoot)[(*outRoot).size() - 1 - index.getValue()].getVOrientation() += omega;
            for(unsigned int i=0; i<in.size(); i++)
                out[i]+=rootX.getOrientation().inverseRotate(in[i]);
        }
        else
        {

            (*outRoot)[index.getValue()].getVCenter() += v;
            (*outRoot)[index.getValue()].getVOrientation() += omega;
            for(unsigned int i=0; i<in.size(); i++)
                out[i]+=rootX.getOrientation().inverseRotate(in[i]);
        }
        /*
        break;

        case 1://one value specified : uniform repartition mapping on the input dofs
        std::cout<<"case 1"<<std::endl;
        val = repartition.getValue()[0];
        cpt=0;
        for(unsigned int ito=0;ito<(*outRoot).size();ito++){
          v=Deriv();omega=Deriv();/////////////////
          for(unsigned int i=0;i<val;i++){
            Deriv f = in[cpt];
            v += f;
            omega += cross(rotatedPoints[cpt],f);
            out[cpt]= rootX.getOrientation().inverseRotate(in[cpt]);
            cpt++;
          }
          (*outRoot)[ito].getVCenter() += v;
          (*outRoot)[ito].getVOrientation() += omega;


        }
        break;

        default:
        std::cout<<"case default"<<std::endl;
        if (repartition.getValue().size() != (*outRoot).size()){
          serr<<"Error : mapping dofs repartition is not correct"<<sendl;
          return;
        }

        cpt=0;
        for(unsigned int ito=0;ito<(*outRoot).size();ito++){
          v=Deriv();omega=Deriv();////////////////////////
          for(unsigned int i=0;i<repartition.getValue()[ito];i++){
            Deriv f = in[cpt];
             out[cpt]= rootX.getOrientation().inverseRotate(in[cpt]);
            v += f;
            omega += cross(rotatedPoints[cpt],f);
            cpt++;
          }
          (*outRoot)[ito].getVCenter() += v;
          (*outRoot)[ito].getVOrientation() += omega;

        }
        break;

        }*/
    }


    else
    {
        out.resize(in.size());
        for(unsigned int i=0; i<in.size(); i++)
        {
            out[i] = in[i];
        }
    }
}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::applyJT( typename In::VecConst&  out , const typename Out::VecConst&  in , typename InRoot::VecConst*  outroot)
{
    int outSize=out.size();
    out.resize(in.size() + outSize); // we can accumulate in "out" constraints from several mappings
    if (rootModel)
    {
        int outRootSize = outroot->size();
        outroot->resize(in.size() + outRootSize); // we can accumulate in "out" constraints from several mappings

        /*       switch (repartition.getValue().size())
               {
               case 0:
                 {
         */



        for(unsigned int i=0; i<in.size(); i++)
        {
            Vector v,omega;
            OutConstraintIterator itIn;
            std::pair< OutConstraintIterator, OutConstraintIterator > iter=in[i].data();

            for (itIn=iter.first; itIn!=iter.second; itIn++)
            {
                const unsigned int node_index = itIn->first;// index of the node
                // out = Jt in
                // Jt = [ I     ]
                //      [ -OM^t ]
                // -OM^t = OM^

                const Deriv f = (Deriv) itIn->second;
                v += f;
                omega += cross(rotatedPoints[node_index],f);
                InDeriv f_deform = rootX.getOrientation().inverseRotate(f);
                out[outSize+i].add(node_index,f_deform);

            }

            const InRoot::Deriv result(v, omega);
            if (!indexFromEnd.getValue())
            {
                (*outroot)[outRootSize+i].add(index.getValue(), result);
            }
            else
            {
                (*outroot)[outRootSize+i].add(out.size() - 1 - index.getValue(), result);
            }
            /*
            }
            break;
            }

            case 1://one value specified : uniform repartition mapping on the input dofs
             std::cout<<"DeformableOnRigidFrameMapping:Case 1 not implemented yet"<<std::endl;
             break;

            default:
             std::cout<<"DeformableOnRigidFrameMapping:Defaut case not implemented yet"<<std::endl;
             break;

             */
        }
    }


    else
    {

    }


}

/// Template specialization for 2D rigids
// template<typename real1, typename real2>
// void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::StdRigidTypes<2, real1> >, core::behavior::MechanicalState< defaulttype::StdVectorTypes<defaulttype::Vec<2, real2>, defaulttype::Vec<2, real2>, real2 > > > >::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
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
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
//#endif
//#ifndef SOFA_DOUBLE
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
//#endif
//
//#ifndef SOFA_FLOAT
//#ifndef SOFA_DOUBLE
//template<>
//void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2fTypes >, core::behavior::MechanicalState< defaulttype::Vec2dTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
//template<>
//    void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::Rigid2dTypes >, core::behavior::MechanicalState< defaulttype::Vec2fTypes > > >::applyJT( In::VecConst& out, const Out::VecConst& in );
//#endif
//#endif
/// Template specialization for 2D rigids
// template<typename real1, typename real2>
// void DeformableOnRigidFrameMapping< core::behavior::MechanicalMapping< core::behavior::MechanicalState< defaulttype::StdRigidTypes<2, real1> >, core::behavior::MechanicalState< defaulttype::StdVectorTypes<defaulttype::Vec<2, real2>, defaulttype::Vec<2, real2>, real2 > > > >::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
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
void DeformableOnRigidFrameMapping<BasicMapping>::propagateX()
{
    if (this->fromModel!=NULL && this->toModel->getX()!=NULL && this->fromModel->getX()!=NULL)
        apply(*this->toModel->getX(), *this->fromModel->getX(), (rootModel==NULL ? NULL : rootModel->getX()));


    if( this->f_printLog.getValue())
    {
        serr<<"DeformableOnRigidFrameMapping::propageX processed :"<<sendl;
        if (rootModel!=NULL)
            serr<<"input root: "<<*rootModel->getX();
        serr<<"  - input: "<<*this->fromModel->getX()<<"  output : "<<*this->toModel->getX()<<sendl;
    }


}

template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::propagateXfree()
{
    if (this->fromModel!=NULL && this->toModel->getXfree()!=NULL && this->fromModel->getXfree()!=NULL)
        apply(*this->toModel->getXfree(), *this->fromModel->getXfree(), (rootModel==NULL ? NULL : rootModel->getXfree()));

    if( this->f_printLog.getValue())
    {
        serr<<"DeformableOnRigidFrameMapping::propageXfree processed"<<sendl;
        if (rootModel!=NULL)
            serr<<"input root: "<<*rootModel->getXfree();
        serr<<"  - input: "<<*this->fromModel->getXfree()<<"  output : "<<*this->toModel->getXfree()<<sendl;
    }

}


template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::propagateV()
{
    if (this->fromModel!=NULL && this->toModel->getV()!=NULL && this->fromModel->getV()!=NULL)
        applyJ(*this->toModel->getV(), *this->fromModel->getV(), (rootModel==NULL ? NULL : rootModel->getV()));

    if( this->f_printLog.getValue())
    {
        serr<<"DeformableOnRigidFrameMapping::propagateV processed"<<sendl;
        if (rootModel!=NULL)
            serr<<"V input root: "<<*rootModel->getV();
        serr<<"  - V input: "<<*this->fromModel->getV()<<"   V output : "<<*this->toModel->getV()<<sendl;
    }

}



template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::propagateDx()
{
    if (this->fromModel!=NULL && this->toModel->getDx()!=NULL && this->fromModel->getDx()!=NULL)
        applyJ(*this->toModel->getDx(), *this->fromModel->getDx(), (rootModel==NULL ? NULL : rootModel->getDx()));


    if( this->f_printLog.getValue())
    {
        serr<<"DeformableOnRigidFrameMapping::propagateDx processed"<<sendl;
        if (rootModel!=NULL)
            serr<<"input root: "<<*rootModel->getDx();
        serr<<"  - input: "<<*this->fromModel->getDx()<<"  output : "<<*this->toModel->getDx()<<sendl;
    }

}



template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::accumulateForce()
{
    if (this->fromModel!=NULL && this->toModel->getF()!=NULL && this->fromModel->getF()!=NULL)
        applyJT(*this->fromModel->getF(), *this->toModel->getF(), (rootModel==NULL ? NULL : rootModel->getF()));


    if( this->f_printLog.getValue())
    {
        serr<<"DeformableOnRigidFrameMapping::accumulateForce processed"<<sendl;
        serr<<" input f : "<<*this->toModel->getF();
        if (rootModel!=NULL)
            serr<<"- output root: "<<*rootModel->getF();
        serr<<"  - output F: "<<*this->fromModel->getF()<<sendl;
    }

}



template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::accumulateDf()
{
    //if (this->fromModel!=NULL && this->toModel->getF()!=NULL && this->fromModel->getF()!=NULL)
    applyJT(*this->fromModel->getF(), *this->toModel->getF(), (rootModel==NULL ? NULL : rootModel->getF()));


    if( this->f_printLog.getValue())
    {
        serr<<"DeformableOnRigidFrameMapping::accumulateDf processed"<<sendl;
        serr<<" input df : "<<*this->toModel->getF();
        if (rootModel!=NULL)
            serr<<"- output root: "<<*rootModel->getF();
        serr<<"  - output: "<<*this->fromModel->getF()<<sendl;
    }

}



template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::accumulateConstraint()
{
    if (this->fromModel!=NULL && this->toModel->getC()!=NULL && this->fromModel->getC()!=NULL)
    {
        //propagateX();
        applyJT(*this->fromModel->getC(), *this->toModel->getC(), (rootModel==NULL ? NULL : rootModel->getC()));

        // Accumulate contacts indices through the MechanicalMapping
        std::vector<unsigned int>::iterator it = this->toModel->getConstraintId().begin();
        std::vector<unsigned int>::iterator itEnd = this->toModel->getConstraintId().end();

        while (it != itEnd)
        {
            this->fromModel->setConstraintId(*it);
            // in case of a "multi-mapping" (the articulation system is placed on a  simulated object)
            // the constraints are transmitted to the rootModle (the <rigidtype> object which is the root of the articulated system)
            if (rootModel!=NULL)
                rootModel->setConstraintId(*it);
            it++;
        }
    }
}

/*
template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::recomputeRigidMass()
{

    if (this->fromModel==NULL || rootModel==NULL)
        return;
    std::cout<<"recmpute Rigid Mass" <<std::endl;


    masses = dynamic_cast<BaseMass*> (this->fromModel->getContext()->getMass());
    if(!masses)
        return;

    totalMass = 0.0;
    //compute the total mass of the object
    for (unsigned int i=0 ; i<this->fromModel->getX()->size() ; i++)
        totalMass += masses->getElementMass(i);







    sofa::core::objectmodel::
    this->fromModel->getContext()->get(rootModel, core::objectmodel::BaseContext::SearchUp);


}
*/



template <class BasicMapping>
void DeformableOnRigidFrameMapping<BasicMapping>::draw()
{
    if (!this->getShow()) return;
    std::vector< Vector3 > points;
    Vector3 point;

    const typename Out::VecCoord& x = *this->toModel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        point = OutDataTypes::getCPos(x[i]);
        points.push_back(point);
    }
    simulation::getSimulation()->DrawUtility.drawPoints(points, 7, Vec<4,float>(1,1,0,1));
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
