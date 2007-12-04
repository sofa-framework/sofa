
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
#ifndef SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_INL
#define SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_INL

#include <sofa/component/mapping/LaparoscopicRigidMapping.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/simulation/tree/GrabVisitor.h>
#include <sofa/simulation/tree/DeleteVisitor.h>
#include <sofa/component/forcefield/VectorSpringForceField.h>
#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <string>



namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace sofa::component::collision;
using namespace sofa::simulation::tree;

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::init()
{
    this->BasicMapping::init();
    mstate = getMechanicalState();
}


template <class BasicMapping>
helper::vector< defaulttype::Vec3f > LaparoscopicRigidMapping<BasicMapping>::getGrabPoints()
{
    helper::vector< defaulttype::Vec3f > result;
    if (mstate==NULL) return result;

    const helper::vector< unsigned int> & indices = grab_index.getValue().getArray();

    for (helper::vector< unsigned int >::const_iterator it = indices.begin(); it != indices.end(); ++it)
        result.push_back((*mstate->getX())[(*it)]);

    return result;
}




template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::processGrab()
{
    //Get the coordinates of the points OF THE TOOL that will be used to place the strings
    const helper::vector< defaulttype::Vec3f > grab_points=getGrabPoints();

    GrabGetPointsVisitor getVisitor;
    getVisitor.points =  grab_index.getValue().getArray();
    getVisitor.tool = mstate;

//Get the Root of the simulation
    GNode *root= static_cast<GNode*>(this->getContext());
    while (root->getParent() != NULL) root = static_cast<GNode*>(root->getParent());
    getVisitor.execute(root);

    for (unsigned int i=0; i<getVisitor.collisions.size(); i++)
    {
        //Given a forcefield, get the models involved
        GrabCollisionModelsVisitor getCollisionVisitor;
        getCollisionVisitor.ff = dynamic_cast< sofa::core::componentmodel::behavior::InteractionForceField* >(getVisitor.collisions[i].first);
        getCollisionVisitor.model1 = NULL;
        getCollisionVisitor.model2 = NULL;
        if (getCollisionVisitor.ff==NULL) continue;
        getCollisionVisitor.execute(static_cast<GNode *>(getCollisionVisitor.ff->getContext()) );

        //The tool as only Points as collision model
        //We are using only triangleModel to place the string
        TriangleModel *model=dynamic_cast<TriangleModel*>(getCollisionVisitor.model1);
        if (!model) model = dynamic_cast<TriangleModel*>(getCollisionVisitor.model2);
        if (model == NULL) continue;

        //Get the triangle where to place the string
        Triangle triangle(model, getVisitor.triangle[i]);
        model = triangle.getCollisionModel();

        //Create a mechanical object that will store the position of the point placed on the object grabbed by the tool
        component::MechanicalObject<Vec3Types>* mstate2 = NULL;
        simulation::tree::GNode* child = NULL;
        simulation::tree::GNode* parent = static_cast< GNode * >(model->getContext());
        if (model->getTopology())
        {
            //Create Node with the contact points
            child = new simulation::tree::GNode("contactTool"); nodes.push_back(child);
            parent->addChild(child);
            child->updateContext();
            mstate2 = new component::MechanicalObject<Vec3Types>;
            child->addObject(mstate2);

            typedef mapping::BarycentricMapping<core::componentmodel::behavior::MechanicalMapping<core::componentmodel::behavior::MechanicalState<TriangleModel::DataTypes>, core::componentmodel::behavior::MechanicalState<Vec3Types> > > TriangleMapping;
            typedef mapping::TopologyBarycentricMapper<topology::MeshTopology,TriangleModel::DataTypes, Vec3Types> TriangleMapper;
            TriangleMapper* mapper = new TriangleMapper(model->getTopology());
            TriangleMapping* mapping = new TriangleMapping(model->getMechanicalState(),mstate2,mapper);

            child->addObject(mapping);
            mstate2->resize(1);
            (*mstate2->getX())[0] = getVisitor.collisions[i].second;
            mstate2->init();
            mapper->clear();

            {
                int index = triangle.getIndex();

                if (index < model->getTopology()->getNbTriangles())
                {
                    mapper->createPointInTriangle(getVisitor.collisions[i].second, index, model->getMechanicalState()->getX());
                }
                else
                {
                    mapper->createPointInQuad(getVisitor.collisions[i].second, (index - model->getTopology()->getNbTriangles())/2, model->getMechanicalState()->getX());
                }
            }

            //Add the Spring
            if (mstate2)
            {
                component::forcefield::StiffSpringForceField<Vec3Types>* contactff = new component::forcefield::StiffSpringForceField<Vec3Types>(mstate,mstate2);
                forcefields.push_back(contactff);
                //index1 - index2 - ks - kd - initlen
                contactff->addSpring(getVisitor.index_point[i], 0, 100000000, 0, 0);
                child->addObject(contactff);
            }
        }
    }
}

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::processRelease()
{
    for (unsigned int i=0; i<forcefields.size(); i++)
    {
        forcefields[i]->getContext()->removeObject(forcefields[i]);
        delete forcefields[i];
    }
    for (unsigned int i=0; i<nodes.size(); i++)
    {
        simulation::tree::DeleteVisitor v;
        nodes[i]->execute(v);
        nodes[i]->getParent()->removeChild(nodes[i]);
        delete nodes[i];
    }
    forcefields.clear();
    nodes.clear();
}


template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::grab()
{
    if (!grab_state)
    {
        processGrab();
        //it means, no spring has been set, so no grabing done : as the tool should be in position grab, we prevent a new grab at next activation
        //if you want to grab at once after a grab failure, just uncomment this line
        //     if (forcefields.size() == 0) return;
    }
    else {processRelease();}
    grab_state = !grab_state;
}


template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(1);
    out[0].getOrientation() = in[0].getOrientation() * rotation.getValue();
    out[0].getCenter() = pivot.getValue() + in[0].getOrientation().rotate(Vector3(in[0].getTranslation(),0,0));
}

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& /*in*/ )
{
    out.resize(1);
    out[0].getVOrientation() = Vector3(); //rotation * in[0].getVOrientation();
    out[0].getVCenter() = Vector3(); //in[0].getOrientation().rotate(Vec<3,Real>(in[0].getVTranslation(),0,0));
}

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::applyJT( typename In::VecDeriv& /*out*/, const typename Out::VecDeriv& /*in*/ )
{
}

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::draw()
{

    if (getContext()->getShowBehaviorModels() && mstate != NULL)
    {
        glDisable (GL_LIGHTING);
        glPointSize(10);
        glColor4f (1.0,1.0,0.0,1);
        glBegin (GL_POINTS);

        const helper::vector< defaulttype::Vec3f > grab_points=getGrabPoints();
        for (unsigned int i=0; i<grab_points.size(); i++)
            helper::gl::glVertexT(grab_points[i]);

        glEnd();

    }
    if (!getShow(this)) return;
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
