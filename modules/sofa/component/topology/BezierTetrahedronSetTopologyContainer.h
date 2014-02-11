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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_BEZIERTETRAHEDRONSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_BEZIERTETRAHEDRONSETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class BezierTetrahedronSetTopologyModifier;

using core::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			         PointID;
typedef BaseMeshTopology::EdgeID			            EdgeID;
typedef BaseMeshTopology::TriangleID		         TriangleID;
typedef BaseMeshTopology::TetraID			         TetraID;
typedef BaseMeshTopology::Edge				         Edge;
typedef BaseMeshTopology::Triangle			         Triangle;
typedef BaseMeshTopology::Tetra				         Tetra;
typedef BaseMeshTopology::SeqTetrahedra			   SeqTetrahedra;


typedef Tetra			Tetrahedron;
typedef EdgesInTetrahedron		EdgesInTetrahedron;
typedef TrianglesInTetrahedron	TrianglesInTetrahedron;
typedef sofa::helper::vector<PointID>         VecPointID;
typedef sofa::helper::vector<TetraID>         VecTetraID;
typedef unsigned char BezierDegreeType;
typedef sofa::helper::fixed_array<BezierDegreeType,4> TetrahedronBezierIndex;
typedef sofa::helper::fixed_array<int,4> ElementTetrahedronIndex;

/** a class that stores a set of Bezier tetrahedra and provides access with adjacent triangles, edges and vertices 
A Bezier Tetrahedron has exactly the same topology as a Tetrahedron but with additional (control) points on its edges, triangles and inside 
We use a Vec4D to number the control points inside  a Bezier tetrahedron */
class SOFA_BASE_TOPOLOGY_API BezierTetrahedronSetTopologyContainer : public TetrahedronSetTopologyContainer
{
    friend class BezierTetrahedronSetTopologyModifier;
	friend class Mesh2BezierTopologicalMapping;

public:
    SOFA_CLASS(BezierTetrahedronSetTopologyContainer,TetrahedronSetTopologyContainer);

    typedef Tetra			Tetrahedron;
    typedef EdgesInTetrahedron		EdgesInTetrahedron;
    typedef TrianglesInTetrahedron	TrianglesInTetrahedron;
protected:
    BezierTetrahedronSetTopologyContainer();

    virtual ~BezierTetrahedronSetTopologyContainer() {}
public:
    virtual void init();
	// build some maps specific of the degree of the tetrahedral elements.
	virtual void reinit();

    /// Bezier Specific Information Topology API
    /// @{
protected :
	/// the degree of the Bezier Tetrahedron 1=linear, 2=quadratic...
	Data <BezierDegreeType> d_degree;
	/// the number of control points corresponding to the vertices of the tetrahedra (different from the total number of points)
    Data<size_t> d_numberOfTetrahedralPoints;
public :
	// specifies where a Bezier Point can lies with respect to the underlying tetrahedral mesh
	enum BezierTetrahedronPointLocation
    {
        POINT = 0,
        EDGE =1 ,
        TRIANGLE = 2,
        TETRAHEDRON = 3
    };
	/// get the Degree of the Bezier Tetrahedron 
	BezierDegreeType getDegree() const;
	/// get the number of control points corresponding to the vertices of the tetrahedra 
	size_t getNumberOfTetrahedralPoints() const;
	/// get the global index of the Bezier  point associated with a given tetrahedron index and given its 4D Index   
	size_t getGlobalIndexOfBezierPoint(const size_t tetrahedronIndex,const TetrahedronBezierIndex id) ;
	/// get the indices of all control points associated with a given tetrahedron
	void getGlobalIndexArrayOfBezierPointsInTetrahedron(const size_t tetrahedronIndex, VecPointID & indexArray) ;
	/// return the Bezier index given the local index in a tetrahedron
	TetrahedronBezierIndex getTetrahedronBezierIndex(const size_t localIndex) const;
	sofa::helper::vector<TetrahedronBezierIndex> getTetrahedronBezierIndexArray() const;
	/// return the local index in a tetrahedron from a tetrahedron Bezier index (inverse of getTetrahedronBezierIndex())
	size_t getLocalIndexFromTetrahedronBezierIndex(const TetrahedronBezierIndex id) const;
	/// return the location, the element index and offset from the global index of a point
	void getLocationFromGlobalIndex(const size_t globalIndex, BezierTetrahedronPointLocation &location, 
		size_t &elementIndex, size_t &elementOffset) ;
	/// check the Bezier Point Topology
	bool checkBezierPointTopology();
	 /// @}

    inline friend std::ostream& operator<< (std::ostream& out, const BezierTetrahedronSetTopologyContainer& t)
    {
        helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = t.d_tetrahedron;
        out  << m_tetrahedron<< " "
                << t.m_edgesInTetrahedron<< " "
                << t.m_trianglesInTetrahedron;

        out << " "<< t.m_tetrahedraAroundVertex.size();
        for (unsigned int i=0; i<t.m_tetrahedraAroundVertex.size(); i++)
        {
            out << " " << t.m_tetrahedraAroundVertex[i];
        }
        out <<" "<< t.m_tetrahedraAroundEdge.size();
        for (unsigned int i=0; i<t.m_tetrahedraAroundEdge.size(); i++)
        {
            out << " " << t.m_tetrahedraAroundEdge[i];
        }
        out <<" "<< t.m_tetrahedraAroundTriangle.size();
        for (unsigned int i=0; i<t.m_tetrahedraAroundTriangle.size(); i++)
        {
            out << " " << t.m_tetrahedraAroundTriangle[i];
        }
		out << " " << t.d_degree.getValue();
		out << " " << t.getNbPoints();
        return out;
    }

    inline friend std::istream& operator>>(std::istream& in, BezierTetrahedronSetTopologyContainer& t)
    {
        unsigned int s;
        sofa::helper::vector< unsigned int > value;
        helper::WriteAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = t.d_tetrahedron;

        in >> m_tetrahedron >> t.m_edgesInTetrahedron >> t.m_trianglesInTetrahedron;


        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_tetrahedraAroundVertex.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_tetrahedraAroundEdge.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_tetrahedraAroundTriangle.push_back(value);
        }
		BezierDegreeType bdt;
		in >> bdt;
		t.d_degree.setValue(bdt);
		int nbp;
		in >> nbp;
		t.setNbPoints(nbp);
        return in;
    }
protected:
	/// Map which provides the location (point, edge, triangle, tetrahedron) of a control point given its tetrahedron Bezier index
	std::map<TetrahedronBezierIndex,ElementTetrahedronIndex> elementMap;
	/// Map which provides the offset in the DOF vector for a control point lying on an edge 
	std::map<TetrahedronBezierIndex,size_t> edgeOffsetMap;
	/// Map which provides the offset in the DOF vector for a control point lying on a triangle 
	std::map<TetrahedronBezierIndex,size_t> triangleOffsetMap;
	/// Map which provides the offset in the DOF vector for a control point lying on a tetrahedron
	std::map<TetrahedronBezierIndex,size_t> tetrahedronOffsetMap;
	/// Map which provides the rank in a control point from the array outputed by getGlobalIndexArrayOfBezierPointsInTetrahedron (consistent with bezierIndexArray) 
	std::map<TetrahedronBezierIndex,size_t> localIndexMap;
	/// array of the tetrahedron Bezier index outputed by the function getGlobalIndexArrayOfBezierPointsInTetrahedron()
	sofa::helper::vector<TetrahedronBezierIndex> bezierIndexArray;

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
