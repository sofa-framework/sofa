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

#ifndef SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_INL
#define SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_INL

#include <sofa/component/mapping/ArticulatedSystemMapping.h>
#include <sofa/core/objectmodel/BaseContext.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::init()
{
    GNode* context = dynamic_cast<GNode*>(this->fromModel->getContext());
    context->getNodeObject(ahc);
    articulationCenters = ahc->getArticulationCenters();

    OutVecCoord& xto = *this->toModel->getX();
    InVecCoord& xfrom = *this->fromModel->getX();

    context->parent->getNodeObject(rootModel);

    apply(xto, xfrom);
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    //std::cout<<"Apply"<<std::endl;
    // Copy the root position if a rigid root model is present
    if (rootModel)
    {
        //	std::cout << "Root Model Name = " << rootModel->getName() << std::endl;
        out[0] = (*rootModel->getX())[rootModel->getSize()-1];
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

        out[child].getOrientation() = out[parent].getOrientation();
        out[child].getCenter() = out[parent].getCenter() + (*ac)->initTranslateChild(out[parent].getOrientation());

        // The position of the articulation center can be deduced using the 6D position of the parent:
        // only useful for visualisation of the mapping
        (*ac)->globalPosition.setValue(out[parent].getCenter() +
                out[parent].getOrientation().rotate((*ac)->posOnParent.getValue()));

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();

        for (; a != aEnd; a++)
        {
            InCoord value = in[(*a)->articulationIndex.getValue()];
            Vec<3,Real> axis = out[child].getOrientation().rotate((*a)->axis.getValue());

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
            }
        }
    }
    //std::cout<<"EndApply"<<std::endl;
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in, const typename InRoot::VecDeriv* inroot )
{
    //std::cout<<"ApplyJ"<<std::endl;
    OutVecCoord& xto = *this->toModel->getX();
    // Copy the root position if a rigid root model is present
    if (inroot)
    {
        // std::cout << "Root Model Name = " << rootModel->getName() << std::endl;
        out[0] = (*inroot)[inroot->size()-1];
    }
    else
        out[0] = OutDeriv();

    out.clear();
    out.resize(xto.size());

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.begin();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = articulationCenters.end();

    for (; ac != acEnd; ac++)
    {
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        out[child].getVOrientation() += out[parent].getVOrientation();
        Vec<3,OutReal> P = xto[parent].getCenter(); Vec<3,OutReal> C = xto[child].getCenter();
        out[child].getVCenter() = out[parent].getVCenter() + cross(P-C, out[parent].getVOrientation());

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();

        for (; a != aEnd; a++)
        {
            InCoord value = in[(*a)->articulationIndex.getValue()];
            Vec<3,OutReal> axis = xto[parent].getOrientation().rotate((*a)->axis.getValue());

            Vec<3,OutReal> A = (*ac)->globalPosition.getValue();

            if ((*a)->rotation.getValue())
            {
                out[child].getVCenter() += cross(A-C, axis*value.x());
                out[child].getVOrientation() += axis*value.x();
            }
            if ((*a)->translation.getValue())
            {
                out[child].getVCenter() += axis*value.x();
            }

        }
    }
    //std::cout<<"End ApplyJ"<<std::endl;
}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in, typename InRoot::VecDeriv* outroot )
{
    //std::cout<<"ApplyJT"<<std::endl;
    OutVecCoord& xto = *this->toModel->getX();

    OutVecDeriv fObjects6DBuf = in;

    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = articulationCenters.end();
    vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acBegin = articulationCenters.begin();

    while (ac != acBegin)
    {
        ac--;
        int parent = (*ac)->parentIndex.getValue();
        int child = (*ac)->childIndex.getValue();

        fObjects6DBuf[parent].getVCenter() += fObjects6DBuf[child].getVCenter();
        Vec<3,OutReal> P = xto[parent].getCenter(); Vec<3,OutReal> C = xto[child].getCenter();
        fObjects6DBuf[parent].getVOrientation() += fObjects6DBuf[child].getVOrientation() + cross(C-P,  fObjects6DBuf[child].getVCenter());

        Vec<3,Real> A = (*ac)->globalPosition.getValue();
        OutDeriv T;
        T.getVCenter() = fObjects6DBuf[child].getVCenter();
        T.getVOrientation() = fObjects6DBuf[child].getVOrientation() + cross(C-A, fObjects6DBuf[child].getVCenter());

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();

        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.end();
        vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aBegin = articulations.begin();

        while (a != aBegin)
        {
            a--;
            Vec<3,OutReal> axis = xto[parent].getOrientation().rotate((*a)->axis.getValue());

            if ((*a)->rotation.getValue())
            {
                out[(*a)->articulationIndex.getValue()].x() += (Real)dot(axis, T.getVOrientation());
            }
            if ((*a)->translation.getValue())
            {
                out[(*a)->articulationIndex.getValue()].x() += (Real)dot(axis, T.getVCenter());
            }
        }
    }

    if (outroot)
    {
        (*outroot)[outroot->size()-1] += fObjects6DBuf[0];
    }
    //std::cout<<"End ApplyJT"<<std::endl;

}

template <class BasicMapping>
void ArticulatedSystemMapping<BasicMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in, typename InRoot::VecConst* outRoot )
{
    //std::cout<<" ApplyJT const"<<std::endl;

    OutVecCoord& xto = *this->toModel->getX();

    out.resize(in.size());

    if (rootModel!=NULL)
        outRoot->resize(in.size()); // the constraints are all transmitted to the root


    for(unsigned int i=0; i<in.size(); i++)
    {
        for (unsigned int j=0; j<in[i].size(); j++)
        {
            const OutSparseDeriv cIn = in[i][j];
            int childIndex = cIn.index;
            const OutDeriv valueConst = (OutDeriv) cIn.data;
            Vec<3,OutReal> posConst = xto[childIndex].getCenter();
            vector<ArticulatedHierarchyContainer::ArticulationCenter*> ACList = ahc->getAcendantList(childIndex);

            vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator ac = ACList.begin();
            vector<ArticulatedHierarchyContainer::ArticulationCenter*>::const_iterator acEnd = ACList.end();

            for (; ac != acEnd; ac++)
            {
                Vec<3,OutReal> posAc = (*ac)->globalPosition.getValue();
                OutDeriv T;
                T.getVCenter() = valueConst.getVCenter();
                T.getVOrientation() = valueConst.getVOrientation() + cross(posConst - posAc, valueConst.getVCenter());

                vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*> articulations = (*ac)->getArticulations();

                vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator a = articulations.begin();
                vector<ArticulatedHierarchyContainer::ArticulationCenter::Articulation*>::const_iterator aEnd = articulations.end();

                int parent = (*ac)->parentIndex.getValue();

                for (; a != aEnd; a++)
                {
                    Vec<3,OutReal> axis = xto[parent].getOrientation().rotate((*a)->axis.getValue());

                    InSparseDeriv constArt;
                    constArt.index = (*a)->articulationIndex.getValue();
                    if ((*a)->rotation.getValue())
                    {
                        constArt.data = (Real)dot(axis, T.getVOrientation());
                    }
                    if ((*a)->translation.getValue())
                    {
                        constArt.data = (Real)dot(axis, T.getVCenter());
                        //printf("\n weightedNormalArticulation : %f", constArt.data);
                    }
                    out[i].push_back(constArt);
                }
            }

            if (rootModel!=NULL)
            {
                Vec<3,OutReal> posRoot = xto[0].getCenter();
                OutDeriv T;
                T.getVCenter() = valueConst.getVCenter();
                T.getVOrientation() = valueConst.getVOrientation() + cross(posConst - posRoot, valueConst.getVCenter());
                unsigned int indexT = 7; //ALLER CHERCHER CETTE INFO!!
                OutSparseDeriv constraintT(indexT, T);

                (*outRoot)[i].push_back(constraintT);
                std::cout<< "constraintT = data : "<< T << "index : "<< indexT<<std::endl;
            }



        }
    }


    //std::cout<<"End ApplyJT const"<<std::endl;

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
    //if (!this->getShow()) return;
    //OutVecCoord& xto = *this->toModel->getX();
    //glDisable (GL_LIGHTING);
    //glPointSize(2);
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
