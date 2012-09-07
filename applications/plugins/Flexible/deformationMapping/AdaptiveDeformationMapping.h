/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_AdaptiveDeformationMAPPING_H
#define SOFA_COMPONENT_MAPPING_AdaptiveDeformationMAPPING_H

#include "../initFlexible.h"

#include "../deformationMapping/BaseDeformationMapping.inl"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/helper/gl/Color.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <set>
#include <map>
#include <utility>
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>


namespace sofa
{
namespace component
{
namespace mapping
{

using helper::vector;


/** This class manages adaptive dofs
*/

template <class JacobianBlockType>
class AdaptiveDeformationMapping : public BaseDeformationMapping<JacobianBlockType>
{
public:
    typedef BaseDeformationMapping<JacobianBlockType> Inherit;
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(AdaptiveDeformationMapping,JacobianBlockType), SOFA_TEMPLATE(BaseDeformationMapping,JacobianBlockType));

    /** Inherited types    */
    //@{
    typedef typename Inherit::Real Real;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::InVecCoord InVecCoord;
    typedef typename Inherit::InVecDeriv InVecDeriv;
    typedef typename Inherit::InMatrixDeriv InMatrixDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::OutVecCoord OutVecCoord;
    typedef typename Inherit::OutVecDeriv OutVecDeriv;
    typedef typename Inherit::OutMatrixDeriv OutMatrixDeriv;
    enum { spatial_dimensions = Inherit::spatial_dimensions };
    enum { material_dimensions = Inherit::material_dimensions };

    typedef helper::WriteAccessor<Data< InVecCoord > > waInCoord;
    typedef helper::WriteAccessor<Data< InVecDeriv> > waInDeriv;
    typedef helper::ReadAccessor<Data< OutVecCoord > > raOutCoord;
    typedef helper::ReadAccessor<Data< OutVecDeriv > > raOutDeriv;

    typedef typename Inherit::BlockType BlockType;
    typedef typename Inherit::ShapeFunctionType ShapeFunctionType;
    typedef typename Inherit::BaseShapeFunction BaseShapeFunction;
    typedef typename Inherit::VReal VReal;
    typedef typename Inherit::Gradient Gradient;
    typedef typename Inherit::VGradient VGradient;
    typedef typename Inherit::Hessian Hessian;
    typedef typename Inherit::VHessian VHessian;
    typedef typename Inherit::VRef VRef;
    typedef typename Inherit::MaterialToSpatial MaterialToSpatial ;
    typedef typename Inherit::VMaterialToSpatial VMaterialToSpatial;
    typedef typename Inherit::mCoord mCoord;

    typedef helper::WriteAccessor<Data<vector<VRef> > >        wa_index;
    typedef helper::WriteAccessor<Data<vector<VReal> > >       wa_w;
    typedef helper::WriteAccessor<Data<vector<VGradient> > >   wa_dw;
    typedef helper::WriteAccessor<Data<vector<VHessian> > >    wa_ddw;

    typedef typename Inherit::Coord Coord ; // spatial coord
    typedef typename Inherit::VecCoord VecCoord;
    //@}

    bool useAdaptivity; // tells if adaptivity is activated or not (depending on the provided inputs and validity of mapping)

    /** graph types */
    //@{
    typedef typename core::topology::BaseMeshTopology::Edge Edge;
    typedef typename core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef helper::ReadAccessor<Data< SeqEdges > > raEdges;
    Data< SeqEdges > graphEdges;

    enum {Active, Inactive, InsertCandidate, RemoveCandidate };
    struct VertexProperty
    {
        VertexProperty():state(Inactive),depth(-1),insertionTime(0) {}
        unsigned int state;
        int depth;
        Real insertionTime;
    };

    struct EdgeProperty
    {
        EdgeProperty(Real _w=0):w(_w) {}
        Real w;
    };
    typedef boost::adjacency_list<boost::setS, boost::vecS,boost::bidirectionalS, VertexProperty, EdgeProperty > Graph; // setS for edges -> disallows parallel edges

    typedef typename boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    typedef typename boost::graph_traits<Graph>::edge_iterator edge_iter;
    typedef typename boost::graph_traits<Graph>::out_edge_iterator out_edge_iter;
    typedef typename boost::graph_traits<Graph>::in_edge_iterator in_edge_iter;

    typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
    typedef typename boost::graph_traits<Graph>::edge_descriptor edge_descriptor;

    Graph fullGraph;        // contains parent -> child dependences between all dofs
    Graph currentGraph;     // contains parent -> child dependences between active and mapped dofs

    //Data< unsigned int > depth;  unsigned int oldDepth;
    Data< Real> threshI;
    Data< Real> threshR;
    Data< Real> minDuration;
    sofa::core::objectmodel::DataFileName dotFilename;
    Data< bool > showGraph;
    //@}

    virtual void init()
    {
        Inherit::init();

        if(!graphEdges.isSet())   // add: test if in=out
        {
            useAdaptivity=false;
            graphEdges.setDisplayed(false);
            //depth.setDisplayed(false);
            threshI.setDisplayed(false);
            threshR.setDisplayed(false);
            minDuration.setDisplayed(false);
            dotFilename.setDisplayed(false);
            showGraph.setDisplayed(false);
            return;
        }

        useAdaptivity=true;
        this->f_listening.setValue(true); // force listening (for runtime adaptivity)
        graphEdges.setGroup("Adaptivity");
        //depth.setGroup("Adaptivity");
        threshI.setGroup("Adaptivity");
        threshR.setGroup("Adaptivity");
        minDuration.setGroup("Adaptivity");
        dotFilename.setGroup("Adaptivity");
        showGraph.setGroup("Adaptivity");

        ComputefullGraph();
        ComputeWeights();
        update();

        // testing
        //ComputeDepth();
        //oldDepth=depth.getValue();  setDepth(depth.getValue());
        //fullGraph[boost::vertex(0,fullGraph)].state=Active;     fullGraph[boost::vertex(1,fullGraph)].state=Active;  fullGraph[boost::vertex(2,fullGraph)].state=Active;  update(); // test
    }


    void draw(const core::visual::VisualParams* vparams)
    {
        Inherit::draw(vparams);

        if(!useAdaptivity) return;

        if (this->showGraph.getValue())
        {
            helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

            raEdges g(this->graphEdges);
            defaulttype::Vec4f col(.1,.1,1,1);
            //   int maxdepth=0; for (unsigned int i=0;i<depth.size();++i) if(maxdepth<depth[i]) maxdepth=depth[i];
            std::vector<defaulttype::Vector3> edge(2);
            for (unsigned int i=0; i<g.size(); ++i)
            {
                Out::get(edge[0][0],edge[0][1],edge[0][2],out[g[i][0]]);
                Out::get(edge[1][0],edge[1][1],edge[1][2],out[g[i][1]]);

                //     int d=(depth[g[i][0]]>depth[g[i][1]])?depth[g[i][0]]:depth[g[i][1]];
                //     sofa::helper::gl::Color::getHSVA(&col[0],(float)d*360./(float)maxdepth,1.,.8,1.);

                vparams->drawTool()->drawArrow(edge[0],edge[1],.05,col);
                //          vparams->drawTool()->drawLines(edge,2.0,col);
            }
        }
    }


protected:
    AdaptiveDeformationMapping (core::State<In>* from = NULL, core::State<Out>* to= NULL)  : Inherit ( from, to )
        , useAdaptivity(false)
        , graphEdges(initData(&graphEdges,SeqEdges(),"graphEdges","oriented graph connecting parent to child nodes"))
        //, depth(initData(&depth,(unsigned int)0,"depth","depth"))
        , threshI(initData(&threshI,(Real)10,"threshI","threshI"))
        , threshR(initData(&threshR,(Real)0.1,"threshR","threshR"))
        , minDuration(initData(&minDuration,(Real)10,"minDuration","minDuration"))
        , dotFilename( initData(&dotFilename, "dotFilename", "output graph file"))
        , showGraph(initData(&showGraph,false,"showGraph","show graph"))
    {
    }

    virtual ~AdaptiveDeformationMapping()     { }


    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if(!useAdaptivity) return;

        if ( dynamic_cast<simulation::AnimateEndEvent*>(event))
        {
            checkforInsertion();
            checkforRemoval();
            //if(oldDepth!=depth.getValue()) { oldDepth=depth.getValue(); setDepth(depth.getValue()); }
            if(this->graphEdges.isDirty()) std::cout<<"edges dirty"<<std::endl;
        }
    }

    /// a node is inserted based on its acceleration
    void checkforInsertion()
    {
        raOutDeriv  ato(*this->toModel->read(core::VecDerivId::force()));   // TODO: check if accelerations can be retrieved

        Real t=this->getContext()->getTime();
        Real dt=this->getContext()->getDt();

        bool changed=false;
        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first)
        {
            vertex_descriptor d = *vp.first;
            if(fullGraph[d].state==InsertCandidate)
            {
                OutDeriv adt = ato[d]*dt;

                Real E = adt.getVec()*adt.getVec(); // kinetic energy, supposing that the mass is identity (for now..)

                if(E>threshI.getValue()) {fullGraph[d].state=Active; fullGraph[d].insertionTime=t; changed=true;}
            }
        }
        if(changed) update();
    }



    /// a node is removed based on its speed and the difference between their current and mapped position
    void checkforRemoval()
    {
        raOutCoord  xto0(*this->toModel->read(core::ConstVecCoordId::restPosition())),  xto(*this->toModel->read(core::ConstVecCoordId::position()));
        raOutDeriv  vto(*this->toModel->read(core::ConstVecDerivId::velocity()));

        Real t=this->getContext()->getTime();
        Real dtinv=1./this->getContext()->getDt();

        bool changed=false;
        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first)
        {
            vertex_descriptor c = *vp.first;
            if(fullGraph[c].state==RemoveCandidate)
                if(t-fullGraph[c].insertionTime > this->minDuration.getValue()) // check duration
                {
                    // mapping
                    OutCoord mappedPos;
                    BlockType J;
                    for (std::pair<in_edge_iter, in_edge_iter> ei = boost::in_edges(c, fullGraph); ei.first != ei.second; ++ei.first)
                    {
                        vertex_descriptor p = source(*ei.first,fullGraph);
                        J.init( xto0[p],xto0[c],this->f_pos0.getValue()[c],this->f_F0.getValue()[c],fullGraph[*ei.first].w,Gradient(),Hessian());
                        J.addapply(mappedPos,xto[p]);
                    }
                    // comparison
                    OutCoord dxdt = (mappedPos - xto[c])*dtinv;
                    OutDeriv v = vto[c];

                    Real E = dxdt.getVec()*dxdt.getVec() + v.getVec()*v.getVec(); // kinetic energy, supposing that the mass is identity (for now..)

                    if(E<threshR.getValue()) {fullGraph[c].state=Inactive; changed=true;}
                    //if(ato[d].getVCenter().norm()<threshI.getValue()) {fullGraph[d].state=Inactive; changed=true;}
                }
        }
        if(changed) update();
    }



    /// update everything when active nodes have changed
    void update()
    {
        UpdatefullGraphCandidates();
        computeCurrentGraph();
        updateMapping();

        if (this->dotFilename.isSet()) PrintGraph();
    }



    /// update input mechanical states and mapping parameters (indices, weights, jacobians) according to currentGraph
    void updateMapping()
    {
        // retrieve active vertices their index
        typedef std::map<vertex_descriptor, unsigned int> IndMap;
        IndMap indmap;
        unsigned int nbActive = 0;
        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first)
        {
            vertex_descriptor d = *vp.first;
            if(fullGraph[d].state==Active || fullGraph[d].state==RemoveCandidate) indmap[d]=nbActive++;
        }

        // setup input mechanical states = copy subset of output states
        waInCoord  xfrom0(*this->fromModel->write(core::VecCoordId::restPosition())),  xfrom(*this->fromModel->write(core::VecCoordId::position())),  xfromReset(*this->fromModel->write(core::VecCoordId::resetPosition()));
        waInDeriv  vfrom(*this->fromModel->write(core::VecDerivId::velocity()));
        raOutCoord  xto0(*this->toModel->read(core::ConstVecCoordId::restPosition())),  xto(*this->toModel->read(core::ConstVecCoordId::position())),  xtoReset(*this->toModel->read(core::VecCoordId::resetPosition()));
        raOutDeriv  vto(*this->toModel->read(core::ConstVecDerivId::velocity()));

        this->fromModel->resize(0);
        this->fromModel->resize(nbActive);

        for(typename IndMap::iterator it=indmap.begin(); it!=indmap.end(); ++it)
        {
            xfrom0[it->second] = xto0[it->first];
            xfrom[it->second] = xto[it->first];
            vfrom[it->second] = vto[it->first];
        }

        // setup mapping weights
        wa_index index(this->f_index);
        wa_w w(this->f_w);
        wa_dw dw(this->f_dw);
        wa_ddw ddw(this->f_ddw);

        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first)
        {
            vertex_descriptor d = *vp.first;
            if(fullGraph[d].state==Inactive || fullGraph[d].state==InsertCandidate) // inactive nodes -> deformation mapping
            {
                int nbParents=boost::in_degree(d, currentGraph);
                index[d].resize(nbParents); w[d].resize(nbParents); dw[d].resize(nbParents); ddw[d].resize(nbParents);
                unsigned int count=0;
                for (std::pair<in_edge_iter, in_edge_iter> ei = boost::in_edges(d, currentGraph); ei.first != ei.second; ++ei.first)
                {
                    vertex_descriptor p = source(*ei.first,currentGraph);
                    index[d][count]=indmap[p]; w[d][count]=currentGraph[*ei.first].w; dw[d][count].fill(0); ddw[d][count].fill(0);
                    count++;
                }
            }
            else // active nodes -> identity mapping
            {
                index[d].resize(1); w[d].resize(1); dw[d].resize(1); ddw[d].resize(1);
                index[d][0]=indmap[d]; w[d][0]=1; dw[d][0].fill(0); ddw[d][0].fill(0);
            }
        }


        // setup jacobians
        for(unsigned int i=0; i<xto0.size(); i++ )
        {
            unsigned int nbref=index[i].size();
            this->jacobian[i].resize(nbref);
            for(unsigned int j=0; j<nbref; j++ )
            {
                this->jacobian[i][j].init( xfrom0[index[i][j]],xto0[i],this->f_pos0.getValue()[i],this->f_F0.getValue()[i],w[i][j],dw[i][j],ddw[i][j]);
            }
        }
        if(this->assembleJ.getValue()) this->updateJ();
    }



    ///  fill full graph based on input edges
    void ComputefullGraph()
    {
        raEdges edges(this->graphEdges); if(!edges.size()) return;
        fullGraph.clear();

        unsigned int numVerts=0;
        for(unsigned int i=0; i<edges.size(); i++)
        {
            boost::add_edge(edges[i][0], edges[i][1], EdgeProperty(), fullGraph);
            if(numVerts<edges[i][0]+1)  numVerts=edges[i][0]+1;
            if(numVerts<edges[i][1]+1)  numVerts=edges[i][1]+1;
        }
    }



    ///  compute edge weights = parent->child mapping weights
    // TODO: use weights computed from shape functions
    void ComputeWeights()
    {
        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first)
        {
            vertex_descriptor ov = *vp.first;
            int nbParents=boost::in_degree(ov, fullGraph);
            for (std::pair<in_edge_iter, in_edge_iter> ei = boost::in_edges(ov, fullGraph); ei.first != ei.second; ++ei.first)
            {
                fullGraph[*ei.first].w=1./(Real)nbParents;
            }
        }
    }


    ///  compute node depth = maximum distance to root
    void ComputeDepth()
    {
        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first) fullGraph[*vp.first].depth=-1;

        bool stop=false;
        while(!stop)
        {
            bool stop2=true;
            for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first)
            {
                vertex_descriptor ov = *vp.first;
                if(fullGraph[ov].depth==-1)
                {
                    stop2=false;
                    in_edge_iter ei, ei_end;
                    if(!boost::in_degree(ov, fullGraph))  fullGraph[ov].depth=0;
                    else for (boost::tie(ei, ei_end) = boost::in_edges(ov, fullGraph); ei != ei_end; ++ei)
                        {
                            vertex_descriptor iv = source(*ei, fullGraph);
                            if(fullGraph[iv].depth==-1) { fullGraph[ov].depth=-1; break; }
                            else fullGraph[ov].depth=fullGraph[iv].depth+1;
                        }
                }
            }
            stop=stop2;
        }
    }

    ///  set active nodes down to a user defined depth
    void setDepth(const unsigned int d)
    {
        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first)
        {
            vertex_descriptor v = *vp.first;
            if(fullGraph[v].depth<=(int)d) fullGraph[v].state=Active;
            else fullGraph[v].state=Inactive;
        }
        update();
    }


    /// reset candidate states according to active/inactive nodes
    void UpdatefullGraphCandidates()
    {
        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first)
        {
            vertex_descriptor d = *vp.first;
            if(!boost::in_degree(d, fullGraph)) fullGraph[d].state=Active; // roots are always active
            else if(fullGraph[d].state==Active || fullGraph[d].state==RemoveCandidate)       //  RemoveCandidates = active nodes with only inactive children
            {
                fullGraph[d].state=RemoveCandidate;
                for (std::pair<out_edge_iter, out_edge_iter> ei = boost::out_edges(d, fullGraph); ei.first != ei.second; ++ei.first)
                {
                    vertex_descriptor o = target(*ei.first,fullGraph);
                    if(fullGraph[o].state==Active || fullGraph[o].state==RemoveCandidate) fullGraph[d].state=Active;
                }
            }
            else    //  InsertCandidates = inactive nodes with only active parents
            {
                fullGraph[d].state=InsertCandidate;
                for (std::pair<in_edge_iter, in_edge_iter> ei = boost::in_edges(d, fullGraph); ei.first != ei.second; ++ei.first)
                {
                    vertex_descriptor i = source(*ei.first,fullGraph);
                    if(fullGraph[i].state==Inactive || fullGraph[i].state==InsertCandidate) fullGraph[d].state=Inactive;
                }
            }
        }
    }


    /// Contract fullgraph to currentgraph with only two levels: active and mapped nodes
    /// to be called whenever the full graph or node state have changed
    void computeCurrentGraph()
    {
        currentGraph.clear();

        // run visitor on each active node
        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(fullGraph); vp.first != vp.second; ++vp.first)
        {
            vertex_descriptor d = *vp.first;
            if(fullGraph[d].state==Active || fullGraph[d].state==RemoveCandidate) {Real w=1.; this->CurrentGraphCreationVisitor(w,d,d);}
        }

        // copy state (for visu)
        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(currentGraph); vp.first != vp.second; ++vp.first)  currentGraph[*vp.first].state=fullGraph[*vp.first].state;
    }

    void CurrentGraphCreationVisitor(Real &w, const vertex_descriptor parent, const vertex_descriptor current)
    {
        for (std::pair<out_edge_iter, out_edge_iter> ei = boost::out_edges(current, fullGraph); ei.first != ei.second; ++ei.first)
        {
            edge_descriptor e = *ei.first;
            vertex_descriptor v = target(e,fullGraph);
            if(fullGraph[v].state==Inactive || fullGraph[v].state==InsertCandidate)
            {
                Real w2= w * fullGraph[e].w;
                std::pair<edge_descriptor, bool> ep = boost::add_edge(parent,v,EdgeProperty(w2),currentGraph);  // create edge  between active and inactive node
                if(!ep.second) currentGraph[ep.first].w+=w2;      // accumulate weight if edge already present (multiple paths through inactive nodes)
                CurrentGraphCreationVisitor(w2, parent, v); // recursively descend the graph along inactive nodes
            }
        }
    }



    /// export graphs to a dot graphviz file
    void PrintGraph()
    {
        if (!this->dotFilename.isSet()) return;
        std::ofstream dot_file(this->dotFilename.getFullPath().c_str());

        if(!dot_file.is_open()) return;

        PrintGraph(dot_file,fullGraph);
        PrintGraph(dot_file,currentGraph);

        dot_file.close();
        std::cout<<"Adaptive mapping: written "<<this->dotFilename.getFullPath().c_str()<<std::endl;
    }

    void PrintGraph(std::ofstream &dot_file,const Graph &g)
    {
        typename boost::property_map<Graph, boost::vertex_index_t>::type indexmap = get(boost::vertex_index, g);

        dot_file << "digraph D {\n" ;

        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(g); vp.first != vp.second; ++vp.first)
        {
            dot_file << *vp.first << "[ label=\"" << indexmap[*vp.first] /*<< "," << g[*vp.first].depth*/ << "\"";
            if(g[*vp.first].state==Active)  dot_file  <<" shape=\"circle\" style=\"filled\""  << "];\n";
            else if(g[*vp.first].state==RemoveCandidate)  dot_file  <<" shape=\"circle\""  << "];\n";
            else if(g[*vp.first].state==Inactive)  dot_file  <<" shape=\"square\" style=\"filled\""  << "];\n";
            else if(g[*vp.first].state==InsertCandidate)  dot_file  <<" shape=\"square\""  << "];\n";
        }
        for (std::pair<edge_iter, edge_iter> ei = boost::edges(g); ei.first != ei.second; ++ei.first)
        {
            vertex_descriptor u = source(*ei.first, g), v = target(*ei.first, g);
            dot_file << u << " -> " << v << "[label=\"" << g[*ei.first].w << "\"";
            //if (p[v] == u) dot_file << ", color=\"black\"";
            ////else
            dot_file << ", color=\"grey\"";
            dot_file << "]\n";
        }
        dot_file << "}\n\n";

        //        std::cout << "vertices = ";
        //        for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(g); vp.first != vp.second; ++vp.first)  std::cout << indexmap[*vp.first] <<  " ";
        //        std::cout << std::endl;

        //        std::cout << "edges = ";
        //        edge_iter ei, ei_end;
        //        for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) std::cout << "(" << indexmap[source(*ei, g)] << "," << indexmap[target(*ei, g)] << ") ";
        //        std::cout << std::endl;
    }

};


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
