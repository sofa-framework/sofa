#ifndef EXPORT_POV_H
#define EXPORT_POV_H

#include "Topology/generic/attributeHandler.h"
#include "Utils/cgognStream.h"
#include "Algo/Geometry/normal.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace ExportPov
{

template <typename PFP>
void exportTriangleWire(std::ofstream& out,typename PFP::VEC3& p1,typename PFP::VEC3& p2,typename PFP::VEC3& p3, float width)
{
		out << "cylinder { <" << p1[0] << "," << p1[1] << "," << p1[2] << ">, <" << p2[0] << "," << p2[1] << "," << p2[2] << ">, " << width << "}" << std::endl;

		out << "cylinder { <" << p1[0] << "," << p1[1] << "," << p1[2] << ">, <" << p3[0] << "," << p3[1] << "," << p3[2] << ">, " << width << "}" << std::endl;

		out << "cylinder { <" << p3[0] << "," << p3[1] << "," << p3[2] << ">, <" << p2[0] << "," << p2[1] << "," << p2[2] << ">, " << width << "}" << std::endl;
}

template <typename PFP>
void exportTrianglePlain(std::ofstream& out,typename PFP::VEC3& p1,typename PFP::VEC3& p2,typename PFP::VEC3& p3)
{
		out << "triangle {" << std::endl;
		out << "<" << p1[0] << "," << p1[2] << "," << p1[1] << ">," << std::endl;
		out << "<" << p2[0] << "," << p2[2] << "," << p2[1] << ">, " << std::endl;
		out << "<" << p3[0] << "," << p3[2] << "," << p3[1] << "> " << std::endl;
		out << "}" << std::endl;
}

template <typename PFP>
void exportMeshPlain(std::ofstream& out, typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position, const std::string& meshName)
{
	out << "#declare " << meshName << "= union {" << std::endl;

	TraversorF<typename PFP::MAP > travF(map);

	for(Dart d = travF.begin() ; d!= travF.end() ; d = travF.next())
	{
		unsigned int nb = map.faceDegree(d);

		if(nb == 3)
			exportTrianglePlain<PFP>(out,position[d],position[map.phi1(d)],position[map.phi1(map.phi1(d))]);
		else
		{
			out << "polygon{ " << nb+1 << std::endl;
			Dart dd = d;
			do
			{
				out << "<" << position[dd][0] << "," << position[dd][2] << "," << position[dd][1] << ">," << std::endl;
				dd = map.phi1(dd);
			} while(dd!=d);
			out << "<" << position[d][0] << "," << position[d][2] << "," << position[d][1] << ">" << std::endl;
			out << "}" << std::endl;
		}

	}

	out << "}" << std::endl;
}

template <typename PFP>
void export3MeshPlainSmooth(std::ofstream& out, typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position, const std::string& meshName)
{
	typedef typename PFP::VEC3 VEC3;

	out << "#declare " << meshName << "= mesh2 {" << std::endl;

	unsigned int nbDarts = map.getNbDarts() ;

	//vector containing the degree of faces
	std::vector<unsigned int> facesSize ;
	//vector containing the list of index of vertices
	std::vector<std::vector<unsigned int> > facesIdx ;
	facesSize.reserve(nbDarts/3) ;
	facesIdx.reserve(nbDarts/3) ;

	//map : attribute place / index in declaration (for vertices and normals)
	std::map<unsigned int, unsigned int> vIndex ;
	//index : start from 0 and increase (used to ignore the gaps potentially present in the container)
	unsigned int vCpt = 0 ;

	//remember the attribute lines
	std::vector<unsigned int> vertices ;

	std::vector<VEC3> normals ;
	vertices.reserve(nbDarts/6) ;
	normals.reserve(nbDarts/6) ;

	CellMarker<VERTEX> markV(map) ;
	DartMarker markF(map) ;
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{

		if(!markF.isMarked(d) && map.phi3(d)==d)
		{
			markF.markOrbit<FACE>(d) ;
			std::vector<unsigned int> fidx ;
			fidx.reserve(4) ;
			Dart dd = d ;
			do
			{
				unsigned int vNum = map.getEmbedding<VERTEX>(dd) ;
				if(!markV.isMarked(dd))
				{
					markV.mark(dd) ;
					VEC3 norm = Geometry::vertexBorderNormal<PFP>(map,dd,position);

					vIndex[vNum] = vCpt++ ;
					vertices.push_back(vNum) ;

					normals.push_back(norm) ;
				}

				fidx.push_back(vIndex[vNum]) ;

				dd = map.phi1(dd) ;
			} while(dd != d) ;

			facesSize.push_back(map.faceDegree(d)) ;
			facesIdx.push_back(fidx) ;
		}
	}

	//export all vertices
	out << "vertex_vectors {" << std::endl;
	out << vertices.size() << "," << std::endl;
	for(unsigned int i = 0; i < vertices.size(); ++i)
	{
		const VEC3& v = position[vertices[i]] ;
		out << "<" << v[0] << ", " << v[1] << ", " << v[2] << ">"<< std::endl ;
	}
	out << "}" << std::endl;

	//export all normals
	out << "normal_vectors {" << std::endl;
	out << normals.size() << "," << std::endl;
	for(unsigned int i = 0; i < normals.size(); ++i)
	{
		const VEC3& v = normals[i];
		out << "<" << v[0] << ", " << v[1] << ", " << v[2] << ">"<< std::endl ;
	}
	out << "}" << std::endl;

	//export all faces
	out << "face_indices {" << std::endl;
	out << facesSize.size() << "," << std::endl;
	for(unsigned int i = 0; i < facesSize.size(); ++i)
	{
		out << "<" << facesIdx[i][0];
		for(unsigned int j = 1; j < facesIdx[i].size(); ++j)
			out << ", " << facesIdx[i][j] ;
		out << ">" << std::endl ;
	}
	out << "}" << std::endl;

	out << "}" << std::endl;
}

template <typename PFP>
void exportMeshWire(std::ofstream& out, typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position, const std::string& meshName)
{
	out << "#declare " << meshName << "= union {" << std::endl;

	TraversorE<typename PFP::MAP > travE(map);

	for(Dart d = travE.begin() ; d!= travE.end() ; d = travE.next())
	{

		Dart dd = map.phi2(d);

		out << "cylinder{ " << std::endl;
		out << "<" << position[d][0] << "," << position[d][1] << "," << position[d][2] << ">," << std::endl;
		out << "<" << position[dd][0] << "," << position[dd][1] << "," << position[dd][2] << ">," << 0.5 << std::endl;
		out << "}" << std::endl;

	}

	out << "}" << std::endl;
}

template <typename PFP>
bool exportScenePov(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position, const std::string& filename, typename PFP::VEC3 cameraPos, typename PFP::VEC3 cameraLook, typename PFP::VEC3 translate, float angle_X, float angle_Y, float angle_Z)
{
	std::ofstream out(filename.c_str(), std::ios::out);
	if (!out.good())
	{
		CGoGNerr << "(export) Unable to open file " << filename << CGoGNendl;
		return false;
	}

	float angleX = angle_X;
	float angleY = angle_Y;
	float angleZ = angle_Z;

	//define the camera position
	out << "camera { location <" << cameraPos[0] << "," << cameraPos[1] << "," << cameraPos[2] << "> look_at <" << cameraLook[0] << "," << cameraLook[1] << "," << cameraLook[2] <<">}" << std::endl;
	//set a "infinite" plane 
// 	out << "plane { y, -1 pigment { color rgb 1 } }" << std::endl;
	//set a sky sphere
	out << "sphere { <0, 0, 0>, 5000";
	out << "texture{ pigment { color rgb <1, 1, 1>}	finish { ambient 1 diffuse 0 } } }" << std::endl;
	//put some lights
	out << "light_source { <" << cameraPos[0] << "," << cameraPos[1] << "," << cameraPos[2] << "> color rgb 0.45}" << std::endl;
// 	out << "light_source { <-120, -300, -10> color rgb 0.25 }"<< std::endl;
	//set a high quality rendering
	out << "global_settings {" << std::endl;
	out << "radiosity {" << std::endl;
	out << "pretrace_start 0.08 pretrace_end 0.04" << std::endl;
	out << "count 100 nearest_count 10 error_bound 0.15 recursion_limit 1 low_error_factor 0.2 gray_threshold 0.0 minimum_reuse 0.015 brightness 1 adc_bailout 0.01/2 normal off media off}" << std::endl;
	out << "max_trace_level 255}" << std::endl;

	exportMeshPlain<PFP>(out,map,position,"myMesh");

	out << "object {myMesh" << std::endl;
 	out << "translate <" << translate[0] << "," << translate[1] << "," << translate[2] << ">" << std::endl;
 	out << "rotate <" << angleX << "," << angleY << "," << angleZ << "> " << std::endl;
 	out << "texture{ pigment{ color rgb<1.0,1.0,1>} finish { ambient rgb 0.05 brilliance 0.5 } } }" << std::endl;

	out.close();
	return true;
}

template <typename PFP>
bool exportScenePovSmooth(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position, const std::string& filename, typename PFP::VEC3 cameraPos, typename PFP::VEC3 cameraLook, typename PFP::VEC3 translate, float angle_X, float angle_Y, float angle_Z)
{
	std::ofstream out(filename.c_str(), std::ios::out);
	if (!out.good()) {
		CGoGNerr << "(export) Unable to open file " << filename << CGoGNendl;
		return false;
	}

	float angleX = angle_X;
	float angleY = angle_Y;
	float angleZ = angle_Z;

	//define the camera position
	out << "camera { location <" << cameraPos[0] << "," << cameraPos[1] << "," << cameraPos[2] << "> look_at <" << cameraLook[0] << "," << cameraLook[1] << "," << cameraLook[2] <<">}" << std::endl;

	//set a "infinite" plane
// 	out << "plane { y, -1 pigment { color rgb 1 } }" << std::endl;

	//set a sky sphere
	out << "sphere { <0, 0, 0>, 5000";
	out << "texture{ pigment { color rgb <1, 1, 1>}	finish { ambient 1 diffuse 0 } } }" << std::endl;

	//put some lights
	out << "light_source { <" << cameraPos[0] << "," << cameraPos[1] << "," << cameraPos[2] << "> color rgb 0.45}" << std::endl;

	//set a high quality rendering
	out << "global_settings {" << std::endl;
//	out << "radiosity {" << std::endl;
//	out << "pretrace_start 0.08 pretrace_end 0.04" << std::endl;
//	out << "count 300 nearest_count 10 error_bound 0.15 recursion_limit 1 low_error_factor 0.2 gray_threshold 0.0 minimum_reuse 0.015 brightness 1 adc_bailout 0.01/2 normal off media off}" << std::endl;
	out << "max_trace_level 60}" << std::endl;

	export3MeshPlainSmooth<PFP>(out,map,position,"myMesh");

	out << "object {myMesh" << std::endl;
 	out << "translate <" << translate[0] << "," << translate[1] << "," << translate[2] << ">" << std::endl;
 	out << "rotate <" << angleX << "," << angleY << "," << angleZ << "> " << std::endl;
 	out << "double_illuminate" << std::endl;
 	out << "texture{ pigment{ color rgb <0.5,1.0,0.5>} finish { ambient 0.5 roughness 0.2 } } }" << std::endl;

	out.close();
	return true;
}

} // namespace ExportPov

}

} // namespace Algo

} // namespace CGoGN

#endif
