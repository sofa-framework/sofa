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

#ifndef SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_INL
#define SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_INL

#include <sofa/component/mapping/ArticulatedSystemMapping.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/gl/template.h>

#include <sofa/simulation/tree/GNode.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using sofa::simulation::tree::GNode;


template <class BasicMapping>
ArticulatedSystemMapping<BasicMapping>::ArticulatedSystemMapping(In* from, Out* to)
    : Inherit(from, to)
    , rootModel(NULL), ahc(NULL)
    , m_rootModelName(initData(&m_rootModelName, std::string(""), "rootModel", "Root position if a rigid root model is specified."))
{

}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::init()
{
    GNode* context = dynamic_cast<GNode*>(this->fromModel->getContext());
    context->getNodeObject(ahc);
    articulationCenters = ahc->getArticulationCenters();

    OutVecCoord& xto = *this->toModel->getX();
    InVecCoord& xfrom = *this->fromModel->getX();

    ArticulationPos.clear();
    ArticulationAxis.clear();
    ArticulationPos.resize(xfrom.size());
    ArticulationAxis.resize(xfrom.size());

    if (!m_rootModelName.getValue().empty())
    {
        std::vector< std::string > tokens(0);
        std::string str = m_rootModelName.getValue();
        std::string::size_type begin_index = 0;
        std::string::size_type end_index = 0;
        if (rootModel)
        {
            serr << "Root Model found : Name = " << rootModel->getName() << sendl;

        }
        while ( (end_index = str.find("/", begin_index)) != std::string::npos )
        {
            tokens.push_back(str.substr(begin_index, end_index - begin_index));
            begin_index = end_index + 1;
        }

        tokens.push_back(str.substr(begin_index));

        GNode* node = context;

        std::vector< std::string >::iterator it = tokens.begin();
        std::vector< std::string >::iterator itEnd = tokens.end();

        while (it != itEnd)
        {
            if ( it->compare("..") == 0 )
            {
                if (node != 0)
                    node = node->parent;
            }
            else
            {
                if (node != 0)
                    node = static_cast<GNode*>(node->getChild(*it));
            }

            ++it;
        }

        if (node != 0)
            node->getNodeObject(rootModel);

    }
    else
    {
        context->parent->getNodeObject(rootModel);
    }

    CoordinateBuf.clear();
    CoordinateBuf.resize(xfrom.size());
    for (unsigned int c=0; c<xfrom.size(); c++)
    {
        CoordinateBuf[c].x() = 0.0;
    }

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();

    for (; ac != acEnd; ac++)
    {
        (*ac)->OrientationArticulationCenter.clear();
        (*ac)->DisplacementArticulationCenter.clear();
        (*ac)->Disp_Rotation.clear();

        // sout << "(*ac)->OrientationArticulationCenter : " << (*ac)->OrientationArticulationCenter << sendl;
        // todo : warning if a (*a)->articulationIndex.getValue() exceed xfrom size !
    }

    apply(xto, xfrom, (rootModel==NULL ? NULL : rootModel->getX()));

    /*
    OutVecDeriv& vto = *this->toModel->getV();
    InVecDeriv& vfrom = *this->fromModel->getV();
    applyJT(vfrom, vto);
    */
}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::reset()
{
    init();
}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in, const typename InRoot::VecCoord* inroot  )
{
    // Copy the root position if a rigid root model is present
    if (rootModel)
    {
        out[0] = (*inroot)[rootModel->getSize()-1];
        //	sout << "Root Model Name = " << rootModel->getName() << sendl;
        //   out[0] = (*rootModel->getX())[rootModel->getSize()-1];
    }

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();

    for (; ac != acEnd; ac++)
    {
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        // Before computing the child position, it is placed with the same orientation than its parent
        // and at the position compatible with the definition of the articulation center
        // (see initTranslateChild function for details...)
        Quat quat_child_buf = out[child].getOrientation();



        // The position of the articulation center can be deduced using the 6D position of the parent:
        // only useful for visualisation of the mapping => NO ! Used in applyJ and applyJT
        (*ac)->globalPosition.setValue(out[parent].getCenter() +
                out[parent].getOrientation().rotate((*ac)->posOnParent.getValue()));

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();

        int process = (*ac)->articulationProcess.getValue();

        switch(process)
        {
        case 0: // 0-(default) articulation are treated one by one, the axis of the second articulation is updated by the potential rotation of the first articulation
            //			   potential problems could arise when rotation exceed 90° (known problem of euler angles)
        {
            // the position of the child is reset to its rest position (based on the postion of the articulation center)
            out[child].getOrientation() = out[parent].getOrientation();
            out[child].getCenter() = out[parent].getCenter() + (*ac)->initTranslateChild(out[parent].getOrientation());

            Vec<3,OutReal> APos;
            APos = (*ac)->globalPosition.getValue();
            for (; a != aEnd; a++)
            {
                int ind = (*a)->articulationIndex.getValue();
                InCoord value = in[ind];
                Vec<3,Real> axis = out[child].getOrientation().rotate((*a)->axis.getValue());
                ArticulationAxis[ind] = axis;

                if ((*a)->rotation.getValue())
                {
                    Quat dq;
                    dq.axisToQuat(axis, value.x());
                    out[child].getCenter() += (*ac)->translateChild(dq, out[child].getOrientation());
                    out[child].getOrientation() += dq;

                }
                if ((*a)->translation.getValue())
                {
                    out[child].getCenter() += axis*value.x();
                    APos += axis*value.x();
                }

                ArticulationPos[ind]= APos;
            }
            break;
        }
        case 1: // the axis of the articulations are linked to the parent - rotations are treated by successive increases -
        {
            //sout<<"Case 1"<<sendl;
            // no reset of the position of the child its position is corrected at the end to respect the articulation center.

            for (; a != aEnd; a++)
            {
                int ind = (*a)->articulationIndex.getValue();
                InCoord value = in[ind];
                InCoord prev_value = CoordinateBuf[ind];
                Vec<3,Real> axis = out[parent].getOrientation().rotate((*a)->axis.getValue());
                ArticulationAxis[ind]=axis;

                // the increment of rotation and translation are stored in dq and disp
                if ((*a)->rotation.getValue() )
                {
                    Quat r;
                    r.axisToQuat(axis, value.x() - prev_value.x());
                    // add the contribution into the quaternion that provides the actual orientation of the articulation center
                    (*ac)->OrientationArticulationCenter+=r;

                }
                if ((*a)->translation.getValue())
                {
                    (*ac)->DisplacementArticulationCenter+=axis*(value.x() - prev_value.x());
                }

            }

            //// in case 1: the rotation of the axis of the articulation follows the parent -> translation are treated "before":


            // step 1: compute the new position of the articulation center and the articulation pos
            //         rq: the articulation center folows the translations
            (*ac)->globalPosition.setValue(out[parent].getCenter() + out[parent].getOrientation().rotate((*ac)->posOnParent.getValue()) + (*ac)->DisplacementArticulationCenter);
            vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();

            for (; a != aEnd; a++)
            {
                Vec<3,OutReal> APos;
                APos = (*ac)->globalPosition.getValue();
                ArticulationPos[(*a)->articulationIndex.getValue()]=APos;
            }

            // step 2: compute the position of the child
            out[child].getOrientation() = out[parent].getOrientation() + (*ac)->OrientationArticulationCenter;
            out[child].getCenter() =  (*ac)->globalPosition.getValue() - out[child].getOrientation().rotate( (*ac)->posOnChild.getValue() );

            break;

        }


        case 2: // the axis of the articulations are linked to the child (previous pos) - rotations are treated by successive increases -
        {
            //sout<<"Case 2"<<sendl;
            // no reset of the position of the child its position is corrected at the end to respect the articulation center.
            //Quat dq(0,0,0,1);
            Vec<3,Real> disp(0,0,0);


            for (; a != aEnd; a++)
            {
                int ind = (*a)->articulationIndex.getValue();
                InCoord value = in[ind];
                InCoord prev_value = CoordinateBuf[ind];
                Vec<3,Real> axis = quat_child_buf.rotate((*a)->axis.getValue());
                ArticulationAxis[ind]=axis;





                // the increment of rotation and translation are stored in dq and disp
                if ((*a)->rotation.getValue() )
                {
                    Quat r;
                    r.axisToQuat(axis, value.x() - prev_value.x());
                    // add the contribution into the quaternion that provides the actual orientation of the articulation center
                    (*ac)->OrientationArticulationCenter+=r;
                }
                if ((*a)->translation.getValue())
                {
                    disp += axis*(value.x()) ;

                }

                //// in case 2: the rotation of the axis of the articulation follows the child -> translation are treated "after"
                //// ArticulationPos do not move
                Vec<3,OutReal> APos;
                APos = (*ac)->globalPosition.getValue();
                ArticulationPos[(*a)->articulationIndex.getValue()]=APos;

            }
            (*ac)->DisplacementArticulationCenter=disp;

            out[child].getOrientation() = out[parent].getOrientation() + (*ac)->OrientationArticulationCenter;
            out[child].getCenter() =  (*ac)->globalPosition.getValue() - out[child].getOrientation().rotate((*ac)->posOnChild.getValue());
            out[child].getCenter() += (*ac)->DisplacementArticulationCenter;

            break;

        }

        }
    }

    //////////////////// buf the actual position of the articulations ////////////////////

    CoordinateBuf.clear();
    CoordinateBuf.resize(in.size());
    for (unsigned int c=0; c<in.size(); c++)
    {
        CoordinateBuf[c].x() = in[c].x();
    }


}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in, const typename InRoot::VecDeriv* inroot )
{

    //sout<<" \n ApplyJ ";
    OutVecCoord& xto = *this->toModel->getX();

    out.clear();
    out.resize(xto.size());

    // Copy the root position if a rigid root model is present
    if (inroot)
    {
        // sout << "Root Model Name = " << rootModel->getName() << sendl;
        out[0] = (*inroot)[inroot->size()-1];
    }
    else
        out[0] = OutDeriv();



    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();

    int i = 0;

    for (; ac != acEnd; ac++)
    {
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        out[child].getVOrientation() += out[parent].getVOrientation();
        Vec<3,OutReal> P = xto[parent].getCenter();
        Vec<3,OutReal> C = xto[child].getCenter();
        out[child].getVCenter() = out[parent].getVCenter() + cross(P-C, out[parent].getVOrientation());
        //sout<<"P:"<< P  <<"- C: "<< C;

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();

        for (; a != aEnd; a++)
        {
            int ind = (*a)->articulationIndex.getValue();
            InCoord value = in[ind];
            Vec<3,OutReal> axis = ArticulationAxis[ind];
            Vec<3,OutReal> A = ArticulationPos[ind];


            if ((*a)->rotation.getValue())
            {
                out[child].getVCenter() += cross(A-C, axis*value.x());
                out[child].getVOrientation() += axis*value.x();
            }
            if ((*a)->translation.getValue())
            {
                out[child].getVCenter() += axis*value.x();
            }
            i++;

        }
    }
}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in, typename InRoot::VecDeriv* outroot )
{

    //sout<<"\n ApplyJt";
    OutVecCoord& xto = *this->toModel->getX();
//	InVecCoord &xfrom= *this->fromModel->getX();

    //apply(xto,xfrom);


    OutVecDeriv fObjects6DBuf = in;

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.end();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acBegin = articulationCenters.begin();

    int i=ArticulationAxis.size();
    while (ac != acBegin)
    {
        ac--;
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        fObjects6DBuf[parent].getVCenter() += fObjects6DBuf[child].getVCenter();
        Vec<3,OutReal> P = xto[parent].getCenter();
        Vec<3,OutReal> C = xto[child].getCenter();
        fObjects6DBuf[parent].getVOrientation() += fObjects6DBuf[child].getVOrientation() + cross(C-P,  fObjects6DBuf[child].getVCenter());


        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.end();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aBegin = articulations.begin();

        while (a != aBegin)
        {
            a--;
            i--;
            int ind = (*a)->articulationIndex.getValue();
            Vec<3,OutReal> axis = ArticulationAxis[ind];
            Vec<3,Real> A = ArticulationPos[ind] ;
            OutDeriv T;
            T.getVCenter() = fObjects6DBuf[child].getVCenter();
            T.getVOrientation() = fObjects6DBuf[child].getVOrientation() + cross(C-A, fObjects6DBuf[child].getVCenter());



            if ((*a)->rotation.getValue())
            {
                out[ind].x() += (Real)dot(axis, T.getVOrientation());
            }
            if ((*a)->translation.getValue())
            {
                out[ind].x() += (Real)dot(axis, T.getVCenter());
            }
        }
    }

    if (outroot)
    {
        (*outroot)[outroot->size()-1] += fObjects6DBuf[0];
    }
}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in, typename InRoot::VecConst* outRoot )
{
//	sout << "ApplyJT const  - size in = " << in.size() << sendl;

    OutVecCoord& xto = *this->toModel->getX();

    out.resize(in.size());
    unsigned int sizeOutRoot =0;

    if (rootModel!=NULL)
    {
        sizeOutRoot = outRoot->size();
        outRoot->resize(in.size() + sizeOutRoot); // the constraints are all transmitted to the root
    }


    for(unsigned int i=0; i<in.size(); i++)
    {
        OutConstraintIterator itOut;
        std::pair< OutConstraintIterator, OutConstraintIterator > iter=in[i].data();

        for (itOut=iter.first; itOut!=iter.second; itOut++)
        {
            int childIndex = itOut->first;
            const OutDeriv valueConst = (OutDeriv) itOut->second;
            Vec<3,OutReal> C = xto[childIndex].getCenter();
            vector<ArticulatedHierarchyContainer::ArticulationCenter*> ACList = ahc->getAcendantList(childIndex);

            vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = ACList.begin();
            vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = ACList.end();


            int ii=0;

            for (; ac != acEnd; ac++)
            {

                vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();
                vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
                vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();


                for (; a != aEnd; a++)
                {
                    int ind=	(*a)->articulationIndex.getValue();
                    InDeriv data;

                    Vec<3,OutReal> axis = ArticulationAxis[ind]; // xto[parent].getOrientation().rotate((*a)->axis.getValue());
                    Vec<3,Real> A = ArticulationPos[ind] ; // Vec<3,OutReal> posAc = (*ac)->globalPosition.getValue();
                    OutDeriv T;
                    T.getVCenter() = valueConst.getVCenter();
                    T.getVOrientation() = valueConst.getVOrientation() + cross(C - A, valueConst.getVCenter());



                    if ((*a)->rotation.getValue())
                    {
                        data = (Real)dot(axis, T.getVOrientation());
                    }
                    if ((*a)->translation.getValue())
                    {
                        data = (Real)dot(axis, T.getVCenter());
                        //printf("\n weightedNormalArticulation : %f", constArt.data);
                    }
                    out[i].add(ind,data);
                    ii++;
                }
            }

            if (rootModel!=NULL)
            {
                unsigned int indexT = rootModel->getSize()-1; // On applique sur le dernier noeud
                Vec<3,OutReal> posRoot = xto[indexT].getCenter();
                OutDeriv T;
                T.getVCenter() = valueConst.getVCenter();
                T.getVOrientation() = valueConst.getVOrientation() + cross(C - posRoot, valueConst.getVCenter());

                (*outRoot)[sizeOutRoot+i].add(indexT,T);
                //sout<< "constraintT = data : "<< T << "index : "<< indexT<<sendl;
                //(*outRoot)[i].push_back(constraintT);
                //	sout<< "constraintT = data : "<< T << "index : "<< indexT<<sendl;
            }
        }
    }

//	sout<<"End ApplyJT const"<<sendl;

}


template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::propagateX()
{
    if (this->fromModel!=NULL && this->toModel->getX()!=NULL && this->fromModel->getX()!=NULL)
        apply(*this->toModel->getX(), *this->fromModel->getX(), (rootModel==NULL ? NULL : rootModel->getX()));
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::propagateXfree()
{
    if (this->fromModel!=NULL && this->toModel->getXfree()!=NULL && this->fromModel->getXfree()!=NULL)
        apply(*this->toModel->getXfree(), *this->fromModel->getXfree(), (rootModel==NULL ? NULL : rootModel->getXfree()));
}


template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::propagateV()
{
    if (this->fromModel!=NULL && this->toModel->getV()!=NULL && this->fromModel->getV()!=NULL)
        applyJ(*this->toModel->getV(), *this->fromModel->getV(), (rootModel==NULL ? NULL : rootModel->getV()));
}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::propagateDx()
{
    if (this->fromModel!=NULL && this->toModel->getDx()!=NULL && this->fromModel->getDx()!=NULL)
        applyJ(*this->toModel->getDx(), *this->fromModel->getDx(), (rootModel==NULL ? NULL : rootModel->getDx()));
}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::accumulateForce()
{
    if (this->fromModel!=NULL && this->toModel->getF()!=NULL && this->fromModel->getF()!=NULL)
        applyJT(*this->fromModel->getF(), *this->toModel->getF(), (rootModel==NULL ? NULL : rootModel->getF()));
}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::accumulateDf()
{
    if (this->fromModel!=NULL && this->toModel->getF()!=NULL && this->fromModel->getF()!=NULL)
        applyJT(*this->fromModel->getF(), *this->toModel->getF(), (rootModel==NULL ? NULL : rootModel->getF()));
}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::accumulateConstraint()
{
    if (this->fromModel!=NULL && this->toModel->getC()!=NULL && this->fromModel->getC()!=NULL)
    {
        applyJT(*this->fromModel->getC(), *this->toModel->getC(), (rootModel==NULL ? NULL : rootModel->getC()));

        // Accumulate contacts indices through the MechanicalMapping
        std::vector<unsigned int>::iterator it = this->toModel->getConstraintId().begin();
        std::vector<unsigned int>::iterator itEnd = this->toModel->getConstraintId().end();

        while (it != itEnd)
        {
            this->fromModel->setConstraintId(*it);
            // in case of a "multi-mapping" (the articulation system is placede on a  simulated object)
            // the constraints are transmitted to the rootModle (the <rigidtype> object which is the root of the articulated system)
            if (rootModel!=NULL)
                rootModel->setConstraintId(*it);
            it++;
        }
    }
}



template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::draw()
{

    if (!this->getShow()) return;
    std::vector< Vector3 > points;
    std::vector< Vector3 > pointsLine;

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();
    unsigned int i=0;
    for (; ac != acEnd; ac++)
    {
//		int parent = (*ac)->parentIndex.getValue();
//		int child = (*ac)->childIndex.getValue();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();
        for (; a != aEnd; a++)
        {

            // Articulation Pos and Axis are based on the configuration of the parent
            int ind= (*a)->articulationIndex.getValue();
            points.push_back(ArticulationPos[ind]);

            pointsLine.push_back(ArticulationPos[ind]);
            Vec<3,OutReal> Pos_axis = ArticulationPos[ind] + ArticulationAxis[ind];
            pointsLine.push_back(Pos_axis);

            i++;
        }
    }

    simulation::getSimulation()->DrawUtility.drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    simulation::getSimulation()->DrawUtility.drawLines(pointsLine, 1, Vec<4,float>(0,0,1,1));





    //
    //OutVecCoord& xto = *this->toModel->getX();
    //glDisable (GL_LIGHTING);
    //glPointSize(2);
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
