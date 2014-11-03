#ifndef GIDMESHLOADER_H
#define GIDMESHLOADER_H

#include <sofa/core/loader/MeshLoader.h>
#include <sofa/core/topology/Topology.h>


namespace sofa
{

namespace component
{

namespace loader
{

class GIDMeshLoader : public sofa::core::loader::MeshLoader
{
public :
	SOFA_CLASS(GIDMeshLoader, sofa::core::loader::MeshLoader);

    typedef sofa::core::topology::Topology::Edge Edge;
    typedef sofa::core::topology::Topology::Triangle Triangle;
    typedef sofa::core::topology::Topology::Quad Quad;
    typedef sofa::core::topology::Topology::Tetrahedron Tetrahedron;
    typedef sofa::core::topology::Topology::Hexahedron Hexahedron;
    typedef sofa::defaulttype::Vector3 Coord;


public :
	virtual bool load();

	template <class T>
	static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
	{
		return BaseLoader::canCreate (obj, context, arg);
	}

protected :
	enum ElementType{ LINEAR, TRIANGLE, QUADRILATERAL, TETRAHEDRA, HEXAHEDRA, PRISM, PYRAMID, SPHERE, CIRCLE };

	GIDMeshLoader();
	virtual ~GIDMeshLoader();

	bool readGID(std::ifstream& file);

private :

	bool readLinearElements(std::ifstream& file);
	bool readTriangleElements(std::ifstream& file);
	bool readQuadrilateralElements(std::ifstream& file);
	bool readTetrahedralElements(std::ifstream& file);
	bool readHexahedralElements(std::ifstream& file);


protected :
//	Data<helper::vector<Coord> > m_vertices;
//	Data<helper::vector<Edge> > m_edges;
//	Data<helper::vector<Triangle> > m_triangles;
//	Data<helper::vector<Quad> > m_quads;
//	Data<helper::vector<Tetrahedron> > m_tetrahedra;
//	Data<helper::vector<Hexahedron> > m_hexahedra;

private :
	unsigned short m_dimensions;
	ElementType m_eltType;
	unsigned short m_nNode;


};

} //namespace loader

} // namespace component

} // namespace sofa

#endif // GIDMESHLOADER_H
