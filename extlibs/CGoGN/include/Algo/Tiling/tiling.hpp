
namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Tilings
{

template <typename PFP>
Tiling<PFP>::Tiling(const Tiling<PFP>& t1, const Tiling<PFP> t2):
    m_map(t1.m_map),
	m_nx(-1),
	m_ny(-1),
	m_nz(-1)
{
	if (&(t1.m_map) != &(t2.m_map))
		CGoGNerr << "Warning, can not merge to Polyhedrons of different maps" << CGoGNendl;

    m_tableVertDarts.reserve(t1.m_tableVertDarts.size() + t2.m_tableVertDarts.size()); // can be too much but ...

    VEC3 center(0);

//    for(typename std::vector<Dart>::const_iterator di = t1.m_tableVertDarts.begin(); di != t1.m_tableVertDarts.end(); ++di)
//    {
//        m_tableVertDarts.push_back(*di);
//        center += m_positions[*di];
//    }

//    // O(n2) pas terrible !!
//    for(typename std::vector<Dart>::const_iterator di = t2.m_tableVertDarts.begin(); di != t2.m_tableVertDarts.end(); ++di)
//    {
//        unsigned int em = m_map.template getEmbedding<VERTEX>(*di);

//        typename std::vector<Dart>::const_iterator dj=t1.m_tableVertDarts.begin();
//        bool found = false;
//        while ((dj !=t1.m_tableVertDarts.end()) && (!found))
//        {
//            unsigned int xm = m_map.template getEmbedding<VERTEX>(*dj);
//            if (xm == em)
//                found = true;
//            else
//                ++dj;
//        }
//        if (!found)
//        {
//            m_tableVertDarts.push_back(*di);
//            center += m_positions[*di];
//        }
//    }

    m_center = center / typename PFP::REAL(m_tableVertDarts.size());
}

template <typename PFP>
void Tiling<PFP>::computeCenter(VertexAttribute<VEC3, MAP>& position)
{
    typename PFP::VEC3 center(0);

    for(typename std::vector<Dart>::iterator di = m_tableVertDarts.begin(); di != m_tableVertDarts.end(); ++di)
    {
        center += position[*di];
    }

    m_center = center / typename PFP::REAL(m_tableVertDarts.size());
}


template <typename PFP>
//void Tiling<PFP>::transform(float* matrice)
void Tiling<PFP>::transform(VertexAttribute<VEC3, MAP>& position, const Geom::Matrix44f& matrice)
{
//	Geom::Vec4f v1(matrice[0],matrice[4],matrice[8], matrice[12]);
//	Geom::Vec4f v2(matrice[1],matrice[5],matrice[9], matrice[13]);
//	Geom::Vec4f v3(matrice[2],matrice[6],matrice[10],matrice[14]);
//	Geom::Vec4f v4(matrice[3],matrice[7],matrice[11],matrice[15]);

    for(typename std::vector<Dart>::iterator di = m_tableVertDarts.begin(); di != m_tableVertDarts.end(); ++di)
    {

        typename PFP::VEC3& pos = position[*di];
//
//		Geom::Vec4f VA(pos[0],pos[1],pos[2],1.0f);
//
//		Geom::Vec4f VB((VA*v1),(VA*v2),(VA*v3),(VA*v4));
//		VEC3 newPos(VB[0]/VB[3],VB[1]/VB[3],VB[2]/VB[3]);

        pos = Geom::transform(pos, matrice);
    }

    // transform the center only in the surface case
    //m_center = Geom::transform(m_center, matrice);

}

template <typename PFP>
void Tiling<PFP>::mark(CellMarker<MAP, VERTEX>& m)
{
    for(typename std::vector<Dart>::iterator di = m_tableVertDarts.begin(); di != m_tableVertDarts.end(); ++di)
    {
        m.mark(*di);
    }
}

template <typename PFP>
bool Tiling<PFP>::exportPositions(const VertexAttribute<VEC3, MAP>& position, const char* filename)
{
	// open file
	std::ofstream out ;
	out.open(filename, std::ios::out) ;

	if (!out.good())
	{
		CGoGNerr << "Unable to open file " << CGoGNendl ;
		return false ;
	}

	out << (m_nx + 1) << " ";
	out << (m_ny + 1) << " ";
	out << (m_nz + 1) << " ";

	for(std::vector<Dart>::iterator it = m_tableVertDarts.begin() ; it != m_tableVertDarts.end() ; ++it)
	{
		out << position[*it];
	}

	return true;
}

} // namespace Tilings

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
