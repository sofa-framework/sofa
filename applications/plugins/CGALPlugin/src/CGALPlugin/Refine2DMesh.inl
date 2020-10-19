/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <CGALPlugin/Refine2DMesh.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/types/RGBAColor.h>

#define CGAL_MESH_2_VERBOSE

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_plus_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Triangulation_vertex_base_with_id_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
// IO
#include <CGAL/IO/Polyhedron_iostream.h>


//CGAL
struct K: public CGAL::Exact_predicates_inexact_constructions_kernel {};

using namespace sofa;

namespace cgal
{
	
	template <class DataTypes>
	Refine2DMesh<DataTypes>::Refine2DMesh()
	: d_points( initData (&d_points, "inputPoints", "Position coordinates (3D, z=0)"))
	, d_edges(initData(&d_edges, "inputEdges", "Constraints (edges)"))
	, d_edgesData1(initData(&d_edgesData1, "inputEdgesData1", "Data values defined on constrained edges"))
	, d_edgesData2(initData(&d_edgesData2, "inputEdgesData2", "Data values defined on constrained edges"))
	, d_seedPoints( initData (&d_seedPoints, "seedPoints", "Seed Points (3D, z=0)") )
	, d_regionPoints( initData (&d_regionPoints, "regionPoints", "Region Points (3D, z=0)") )
	, d_useInteriorPoints( initData (&d_useInteriorPoints, true, "useInteriorPoints", "should inputs points not on boundaries be input to the meshing algorithm"))
	, d_newPoints( initData (&d_newPoints, "outputPoints", "New Positions coordinates (3D, z=0)") )
	, d_newTriangles(initData(&d_newTriangles, "outputTriangles", "List of triangles"))
	, d_newEdges(initData(&d_newEdges, "outputEdges", "New constraints (edges)"))
	, d_newEdgesData1(initData(&d_newEdgesData1, "outputEdgesData1", "Data values defined on new constrained edges"))
	, d_newEdgesData2(initData(&d_newEdgesData2, "outputEdgesData2", "Data values defined on new constrained edges"))
	, d_trianglesRegion(initData(&d_trianglesRegion, "trianglesRegion", "Region for each Triangle"))
	, d_newBdPoints(initData(&d_newBdPoints, "outputBdPoints", "Indices of points on the boundary"))
	, p_shapeCriteria(initData(&p_shapeCriteria, 0.125, "shapeCriteria", "Shape Criteria"))
	, p_sizeCriteria(initData(&p_sizeCriteria, 0.5, "sizeCriteria", "Size Criteria"))
	, p_viewSeedPoints(initData(&p_viewSeedPoints, false, "viewSeedPoints", "Display Seed Points"))
	, p_viewRegionPoints(initData(&p_viewRegionPoints, false, "viewRegionPoints", "Display Region Points"))
	{
		
	}
	
	template <class DataTypes>
	void Refine2DMesh<DataTypes>::init()
	{
		addInput(&d_points);
		addInput(&d_edges);
		addInput(&d_edgesData1);
		addInput(&d_edgesData2);
		addInput(&d_seedPoints);
		addInput(&d_regionPoints);
		addInput(&d_useInteriorPoints);
		
		addOutput(&d_newPoints);
		addOutput(&d_newTriangles);
		addOutput(&d_newEdges);
		addOutput(&d_newEdgesData1);
		addOutput(&d_newEdgesData2);
		addOutput(&d_trianglesRegion);
		addOutput(&d_newBdPoints);
		
		setDirtyValue();
	}
	
	template <class DataTypes>
	void Refine2DMesh<DataTypes>::reinit()
	{
        setDirtyValue();
	}
	
	template <class DataTypes>
	void Refine2DMesh<DataTypes>::doUpdate()
	{
		typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
		typedef CGAL::Triangulation_vertex_base_with_id_2<K> Vb;
		typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
		typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
		typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT1;
		typedef CGAL::Constrained_triangulation_plus_2<CDT1>       CDT;
		
		typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
		
		typedef CDT::Constraint_id Constraint_id;
		typedef CDT::Vertex_iterator Vertex_iterator;
		typedef CDT::Face_iterator Face_iterator;
		typedef CDT::Vertex_handle Vertex_handle;
		typedef CDT::Face_handle Face_handle;
		typedef CDT::Point CGALPoint;
		
		
		helper::ReadAccessor< Data< VecCoord > > points = d_points;
		helper::ReadAccessor< Data< VecCoord > > seeds = d_seedPoints;
		helper::ReadAccessor< Data< VecCoord > > regions = d_regionPoints;
		helper::ReadAccessor< Data< SeqEdges > > edges = d_edges;
		helper::ReadAccessor< Data< VecReal > > edgesData1 = d_edgesData1;
		helper::ReadAccessor< Data< VecReal > > edgesData2 = d_edgesData2;
		const bool useInteriorPoints = d_useInteriorPoints.getValue();
		
		helper::WriteAccessor< Data< VecCoord > > newPoints = d_newPoints;
		helper::WriteAccessor< Data< SeqTriangles > > newTriangles = d_newTriangles;
		helper::WriteAccessor< Data< SeqEdges > > newEdges = d_newEdges;
		helper::WriteAccessor< Data< VecReal > > newEdgesData1 = d_newEdgesData1;
		helper::WriteAccessor< Data< VecReal > > newEdgesData2 = d_newEdgesData2;
		helper::WriteAccessor< Data< sofa::helper::vector<int> > > m_tags = d_trianglesRegion;
		helper::WriteAccessor< Data< sofa::helper::vector<PointID> > > newBdPoints = d_newBdPoints;
		
		newPoints.clear();
		newTriangles.clear();
		newEdges.clear();
		newEdgesData1.clear();
		newEdgesData2.clear();
		m_tags.clear();
		newBdPoints.clear();
		
		if (points.empty() && edges.empty())
		return;
		
		CDT cdt;
		std::map<unsigned int, Vertex_handle> mapPointVertexHandle;
		//Insert points
		
		if (useInteriorPoints)
		{
			for (unsigned int i=0 ; i<points.size() ; i++)
			{
				Point p = points[i];
				Vertex_handle vh = cdt.insert(CGALPoint(p[0], p[1]));
				mapPointVertexHandle[i] = vh;
			}
		}
		
		//Insert edges (constraints)
		for (unsigned int i=0 ; i<edges.size() ; i++)
		{
			Edge e = edges[i];
			if (mapPointVertexHandle.find(e[0]) == mapPointVertexHandle.end())
			mapPointVertexHandle[e[0]] = cdt.insert(CGALPoint(points[e[0]][0], points[e[0]][1]));
			if (mapPointVertexHandle.find(e[1]) == mapPointVertexHandle.end())
			mapPointVertexHandle[e[1]] = cdt.insert(CGALPoint(points[e[1]][0], points[e[1]][1]));
			std::pair<Vertex_handle, Vertex_handle> edge (mapPointVertexHandle[e[0]], mapPointVertexHandle[e[1]]);
			cdt.insert(edge.first, edge.second);
		}
		
		//Prepare seed Points
		std::list<CGALPoint> listOfSeeds;
		for (unsigned int i=0 ; i<seeds.size() ; i++)
		{
			Point p = seeds[i];
			listOfSeeds.push_back(CGALPoint(p[0], p[1]));
		}
		
		//Refine
		CGAL::refine_Delaunay_mesh_2(cdt, listOfSeeds.begin(), listOfSeeds.end(), Criteria(p_shapeCriteria.getValue(), p_sizeCriteria.getValue()));
		
		std::map<Vertex_handle, unsigned int> mapping;
		Vertex_iterator vi = cdt.vertices_begin();
		//create indices for points
		newPoints.clear();
		for ( ; vi != cdt.vertices_end(); ++vi)
		{
			CGAL_assertion( ! cdt.is_infinite( vi));
			
			mapping[vi] = (unsigned int)(newPoints.size());
			Point p(CGAL::to_double(vi->point().x()), CGAL::to_double(vi->point().y()), 0.0);
			newPoints.push_back(p);
		}
		
		//////////tag different regions
		m_tags.resize(cdt.number_of_faces());
		//m_tags.fill(-1);
		for(unsigned int i = 0; i < m_tags.size(); ++i)
		m_tags[i] = -1;
		
		//create indices for faces
		std::map<Face_handle, unsigned int> mappingFace;
		unsigned int fn = 0;
		for(Face_iterator fi = cdt.finite_faces_begin(); fi != cdt.finite_faces_end(); ++fi, ++fn)
		{
			mappingFace[fi] = fn;
		}
		
		//for each regionPoint, search the triangle in which the regionPoint is
		std::vector<Face_handle> seedFaces;
		for (unsigned int i = 0; i < regions.size(); ++i)
		{
			CGALPoint p(regions[i][0], regions[i][1]);
			bool inside = false;
			for(Face_iterator fi = cdt.finite_faces_begin(); fi != cdt.finite_faces_end(); ++fi)
			{
				CDT::Triangle tri = cdt.triangle(fi);
				double area0 = CGAL::area(p, tri[0], tri[1]);
				double area1 = CGAL::area(p, tri[1], tri[2]);
				double area2 = CGAL::area(p, tri[2], tri[0]);
				if(area0 > 0.0 && area1 > 0.0 && area2 > 0.0)
				{
					msg_info() << "regionFaces[" << i <<"] = " << mappingFace[fi];
					seedFaces.push_back(fi);
					inside = true;
					break;
				}
			}
			if(!inside)
			msg_warning() << "RegionPoint[" << i << "] isn't inside the mesh.";
			
		}
		
		//for each region triangle, tag the whole region
		for(unsigned int i = 0; i < seedFaces.size(); ++i)
		{
			std::vector<Face_handle> region;
			sofa::helper::vector<bool> flags;
			flags.resize(cdt.number_of_faces());
			flags.fill(0);
			region.push_back(seedFaces[i]);
			flags[mappingFace[seedFaces[i]]] = true;
			
			for(unsigned int j = 0; j < region.size(); ++j)
			{
				Face_handle fh = region[j];
				m_tags[mappingFace[fh]] = i;

				//search neighbors in the same region
				for(unsigned int k = 0; k < 3; ++k)
				{
					Face_handle neighbor = fh->neighbor(k);
					if (cdt.is_infinite(neighbor))
					continue;
					if(flags[mappingFace[neighbor]])
					//already done
					continue;
					if (cdt.is_constrained(*cdt.incident_edges(fh->vertex(cdt.cw(k)), fh)))
					continue;
					flags[mappingFace[neighbor]] = true;
					//if(cdt.are_there_incident_constraints(fh->vertex(cdt.cw(k))) && cdt.are_there_incident_constraints(fh->vertex(cdt.ccw(k))))
					//outside the region
					
					//	continue;
					region.push_back(neighbor);
				}
			}
		}
		
		//////////////////////////////////
		
		//output faces
		Face_iterator fi = cdt.faces_begin();
		for ( ; fi != cdt.faces_end() ; ++fi)
		{
			CGAL_assertion( mapping.find((fi->vertex(0))) != mapping.end());
			CGAL_assertion( mapping.find((fi->vertex(1))) != mapping.end());
			CGAL_assertion( mapping.find((fi->vertex(2))) != mapping.end());
			
			if(m_tags[mappingFace[fi]] == -1)
			continue;
			
			Triangle t (mapping[ (fi->vertex(0))], mapping[ (fi->vertex(1))], mapping[ (fi->vertex(2))]);
			newTriangles.push_back(t);
		}
		
		//output edges
		
		std::set<PointID> bdPoints;
		for (unsigned int i=0 ; i<edges.size() ; i++)
		{
			Edge e = edges[i];
            Vertex_handle va = mapPointVertexHandle[e[0]], vb = mapPointVertexHandle[e[1]];
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(5,0,0)
            Constraint_id cid = cdt.insert(va, vb);

            for (typename CDT::Vertices_in_constraint_iterator it =
                cdt.vertices_in_constraint_begin(cid), succ = it, itend =
                cdt.vertices_in_constraint_end(cid); ++succ != itend; ++it)
#else
            for (typename CDT::Vertices_in_constraint_iterator it =
                cdt.vertices_in_constraint_begin(va, vb), succ = it, itend =
                cdt.vertices_in_constraint_end(va, vb); ++succ != itend; ++it)
#endif
            {
				std::pair<Vertex_handle, Vertex_handle> edge(*it, *succ);
				bool first = true;
				Vertex_handle prev;
				while (edge.first != edge.second)
				{
					std::pair<Vertex_handle, Vertex_handle> edge2 = edge;
					if (!cdt.is_edge(edge2.first, edge2.second))
					{
						Face_handle f; int i;
						if (!cdt.includes_edge(edge.first, edge.second, edge2.second, f, i))
						{
							typename CDT::Vertex_circulator vit = cdt.incident_vertices(edge.first), vitend = vit;
							Coord p0 ( edge2.first->point()[0], edge2.first->point()[1], 0 );
							Coord dir1 ( edge2.second->point()[0] - p0[0], edge2.second->point()[1] - p0[1], 0 );
							dir1.normalize();
							Real best_fit = -1;
							for (int ei=0; ei == 0 || vit != vitend; ++ei, ++vit)
							{
								Vertex_handle v = vit;
								if (!first && prev == v) continue; // do not go back
								Coord dir2 ( v->point()[0] - p0[0], v->point()[1] - p0[1], 0 );
								dir2.normalize();
								Real fit = dir1 * dir2;
								if (fit > best_fit)
								{
									best_fit = fit;
									edge2.second = v;
								}
							}
							if (best_fit < 0)
							{
								serr << "Invalid constrained edge." << sendl;
								break;
							}
						}
					}
					CGAL_assertion( cdt.is_edge(edge2.first, edge2.second));
					CGAL_assertion( mapping.find(edge2.first) != mapping.end());
					CGAL_assertion( mapping.find(edge2.second) != mapping.end());
					bdPoints.insert(mapping[edge2.first]);
					bdPoints.insert(mapping[edge2.second]);
					newEdges.push_back(Edge(mapping[edge2.first], mapping[edge2.second]));
					if (i < edgesData1.size())
					newEdgesData1.push_back(edgesData1[i]);
					if (i < edgesData2.size())
					newEdgesData2.push_back(edgesData2[i]);
					prev = edge.first;
					edge.first = edge2.second;
				}
			}
		}

		/*
		 for (typename CDT::Subconstraint_iterator scit = cdt.subconstraints_begin();
		 scit != cdt.subconstraints_end(); ++scit)
		 {
		 std::pair<Vertex_handle, Vertex_handle> edge = scit->first;
		 CGAL_assertion( mapping.find(edge.first) != mapping.end());
		 CGAL_assertion( mapping.find(edge.second) != mapping.end());
		 newEdges.push_back(Edge(mapping[edge.first], mapping[edge.second]));
		 }
		 */
		
		// output boundary points indices
		for (std::set<PointID>::const_iterator it=bdPoints.begin(), itend=bdPoints.end(); it != itend; ++it)
		newBdPoints.push_back(*it);
	}
	
	template <class DataTypes>
	void Refine2DMesh<DataTypes>::draw(const sofa::core::visual::VisualParams* vparams)
	{
		if (p_viewSeedPoints.getValue())
		{
            vparams->drawTool()->saveLastState();

			const VecCoord& seeds = d_seedPoints.getValue();
            sofa::helper::vector<sofa::defaulttype::Vec3> points;
            sofa::helper::types::RGBAColor color(0.0, 0.0, 1.0, 1);

            for (unsigned int i = 0; i < seeds.size(); i++)
                points.push_back(seeds[i]);
                
            vparams->drawTool()->drawPoints(points, 5, color);
            vparams->drawTool()->restoreLastState();
		}
		
		if (p_viewRegionPoints.getValue())
		{
            vparams->drawTool()->saveLastState();

            const VecCoord& regions = d_regionPoints.getValue();
            sofa::helper::vector<sofa::defaulttype::Vec3> points;
            sofa::helper::types::RGBAColor color(1.0, 0.0, 0.0, 1);

            for (unsigned int i = 0; i < regions.size(); i++)
                points.push_back(regions[i]);

            vparams->drawTool()->drawPoints(points, 5, color);
            vparams->drawTool()->restoreLastState();
		}
	}
	
} //cgal
