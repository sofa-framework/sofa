//
// C++ Implementation: BglNode
//
// Description:
//
//
// Author: Francois Faure in The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "BglNode.h"
#include "BglScene.h"
#include <sofa/core/objectmodel/BaseContext.h>
//#include "bfs_adapter.h"
#include "dfv_adapter.h"
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/topological_sort.hpp>
//#include <boost/property_map.hpp>
#include <boost/vector_property_map.hpp>
#include <iostream>
using std::cerr;
using std::endl;



namespace sofa
{
namespace simulation
{
namespace bgl
{

BglNode::BglNode(BglScene* s, BglScene::Hgraph *g,  BglScene::Hvertex n, const std::string& name)
    : sofa::simulation::Node(name), scene(s), graph(g), vertexId(n)
{
}


BglNode::~BglNode()
{
}


void BglNode::doExecuteVisitor( Visitor* vis )
{
    //cerr<<"BglNode::doExecuteVisitor( simulation::tree::Visitor* action)"<<endl;

    boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(*graph) );
    //boost::queue<BglScene::Hvertex> queue;

    /*    boost::breadth_first_search(
            graph,
            boost::vertex(this->vertexId, *graph),
            queue,
            bfs_adapter(vis,scene->h_vertex_node_map),
            colors
        );*/
    dfv_adapter dfv(vis,scene->h_vertex_node_map);
    boost::depth_first_visit(
        *graph,
        boost::vertex(this->vertexId, *graph),
        dfv,
        colors,
        dfv
    );
}

void BglNode::clearInteractionForceFields()
{
    interactionForceField.clear();
}

namespace
{
struct GetObjectsVisitor: public Visitor
{
    typedef sofa::core::objectmodel::ClassInfo ClassInfo;
    typedef sofa::core::objectmodel::BaseContext::GetObjectsCallBack GetObjectsCallBack;

    const ClassInfo& class_info;
    GetObjectsCallBack& container;

    GetObjectsVisitor(const ClassInfo& class_inf, GetObjectsCallBack& cont)
        : class_info(class_inf), container(cont)
    {}

    Result processNodeTopDown( simulation::Node* node )
    {
        for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
        {
            void* result = class_info.dynamicCast(*it);
            if (result != NULL)
                container(result);
        }
        return RESULT_CONTINUE;
    }
};

}
/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void BglNode::getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir) const
{
    GetObjectsVisitor getobjs(class_info, container);
    if ( dir == SearchDown )
    {
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(scene->hgraph) );
        dfv_adapter dfv( &getobjs, scene->h_vertex_node_map );
        boost::depth_first_visit(
            scene->hgraph,
            boost::vertex(this->vertexId, scene->hgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchUp )
    {
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(scene->rgraph) );
        dfv_adapter dfv( &getobjs, scene->r_vertex_node_map );
        BglScene::Rvertex thisvertex = scene->r_node_vertex_map[scene->h_vertex_node_map[this->vertexId]];
        boost::depth_first_visit(
            scene->rgraph,
            boost::vertex(thisvertex, scene->rgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchRoot )
    {
        scene->dfv( scene->masterVertex, getobjs );
    }
    else    // Local
    {
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            void* result = class_info.dynamicCast(*it);
            if (result != NULL)
                container(result);
        }
    }
}






}
}
}


