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

#ifndef SOFA_COMPONENT_MAPPING_LINESETSKINNINGMAPPING_INL
#define SOFA_COMPONENT_MAPPING_LINESETSKINNINGMAPPING_INL

#include <sofa/component/mapping/LineSetSkinningMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
Vector3 LineSetSkinningMapping<BasicMapping>::projectToSegment(Vector3& first, Vector3& last, OutCoord& vertice)
{
    Vec3d segment,v_f,v_l;

    segment = last - first;
    v_f = vertice-first;
    v_l = vertice-last;

    if(v_f*segment>0.0 && -segment*v_l>0.0)
    {
        double prod = v_f*segment;
        return first + (segment * (prod/segment.norm2()));
    }
    else
    {
        if (v_l.norm() > v_f.norm())
            return first;
        else
            return last;
    }
}

template <class BasicMapping>
double LineSetSkinningMapping<BasicMapping>::convolutionSegment(Vector3& first, Vector3& last, OutCoord& vertice)
{
    int steps = 1000;
    double sum = 0.0;
    Vector3 dist, line;

    line=last-first;

    // False integration
    for(int i=0; i<=steps; i++)
    {
        dist = ((line * i) / steps) + first - vertice;
        sum += pow(1/dist.norm(),weightCoef.getValue());
    }

    sum *= line.norm()/steps;
    return sum;
}

template <class BasicMapping>
void LineSetSkinningMapping<BasicMapping>::init()
{
    OutVecCoord& xto = *this->toModel->getX();
    InVecCoord& xfrom = *this->fromModel->getX();
    core::componentmodel::topology::Topology* topology = dynamic_cast<core::componentmodel::topology::Topology*>(this->fromModel->getContext()->getTopology());
    topology::MeshTopology* t = dynamic_cast<topology::MeshTopology*>(topology);
    linesInfluencedByVertice.resize(xto.size());

    verticesInfluencedByLine.resize(t->getNbLines());

    neighborhoodLinesSet.resize(t->getNbLines());
    neighborhood.resize(t->getNbLines());

    for(unsigned int line1Index=0; line1Index< (unsigned) t->getNbLines(); line1Index++)
    {
        const topology::MeshTopology::Line& line1 = t->getLine(line1Index);
        for(unsigned int line2Index=0; line2Index< (unsigned) t->getNbLines(); line2Index++)
        {
            const topology::MeshTopology::Line& line2 = t->getLine(line2Index);
            if ((line1[0] == line2[0]) || (line1[0] == line2[1]) || (line1[1] == line2[0]))
            {
                neighborhoodLinesSet[line1Index].insert(line2Index);
            }
        }
    }

    for(unsigned int line1Index=0; line1Index< (unsigned) t->getNbLines(); line1Index++)
    {
        std::set<int> result;
        std::insert_iterator<std::set<int> > res_ins(result, result.begin());

        neighborhood[line1Index] = neighborhoodLinesSet[line1Index];

        for(unsigned int i=0; i<nvNeighborhood.getValue()-1; i++)
        {
            for (std::set<int>::const_iterator it = neighborhood[line1Index].begin(), itbegin = it, itend = neighborhood[line1Index].end(); it != itend; it++)
            {
                set_union(itbegin, itend, neighborhoodLinesSet[(*it)].begin(), neighborhoodLinesSet[(*it)].end(), res_ins);
            }

            neighborhood[line1Index] = result;
        }
    }

    for(unsigned int verticeIndex=0; verticeIndex<xto.size(); verticeIndex++)
    {
        double	sumWeights = 0.0;
        vector<influencedLineType> lines;
        lines.resize(t->getNbLines());

        for(unsigned int lineIndex=0; lineIndex< (unsigned) t->getNbLines(); lineIndex++)
        {
            const topology::MeshTopology::Line& line = t->getLine(lineIndex);
            double _weight = convolutionSegment(xfrom[line[0]].getCenter(), xfrom[line[1]].getCenter(), xto[verticeIndex]);

            for(unsigned int lineInfluencedIndex=0; lineInfluencedIndex<lines.size(); lineInfluencedIndex++)
            {
                if(lines[lineInfluencedIndex].weight <= _weight)
                {
                    for(unsigned int i=lines.size()-1; i > lineInfluencedIndex; i--)
                    {
                        lines[i].lineIndex = lines[i-1].lineIndex;
                        lines[i].weight = lines[i-1].weight;
                        lines[i].position = lines[i-1].position;
                    }
                    lines[lineInfluencedIndex].lineIndex = lineIndex;
                    lines[lineInfluencedIndex].weight = _weight;
                    lines[lineInfluencedIndex].position = xfrom[line[0]].getOrientation().inverseRotate(xto[verticeIndex] - xfrom[line[0]].getCenter());
                    break;
                }
            }
        }

        unsigned int lineInfluencedIndex = 0;
        int max = lines[lineInfluencedIndex].lineIndex;
        sumWeights += lines[lineInfluencedIndex].weight;
        linesInfluencedByVertice[verticeIndex].push_back(lines[lineInfluencedIndex]);
        influencedVerticeType vertice;
        vertice.verticeIndex = verticeIndex;
        vertice.weight = lines[lineInfluencedIndex].weight;
        vertice.position = lines[lineInfluencedIndex].position;
        verticesInfluencedByLine[lineInfluencedIndex].push_back(vertice);

        lineInfluencedIndex++;

        while (linesInfluencedByVertice[verticeIndex].size() < numberInfluencedLines.getValue() && lineInfluencedIndex < lines.size())
        {
            if (neighborhood[max].count(lines[lineInfluencedIndex].lineIndex) != 0)
            {
                sumWeights += lines[lineInfluencedIndex].weight;
                linesInfluencedByVertice[verticeIndex].push_back(lines[lineInfluencedIndex]);
                vertice.verticeIndex = verticeIndex;
                vertice.weight = lines[lineInfluencedIndex].weight;
                vertice.position = lines[lineInfluencedIndex].position;
                verticesInfluencedByLine[lineInfluencedIndex].push_back(vertice);
            }
            lineInfluencedIndex++;
        }

        for (unsigned int influencedLineIndex=0; influencedLineIndex<linesInfluencedByVertice[verticeIndex].size(); influencedLineIndex++)
            linesInfluencedByVertice[verticeIndex][influencedLineIndex].weight /= sumWeights;
    }
}

template <class BasicMapping>
void LineSetSkinningMapping<BasicMapping>::reinit()
{
    linesInfluencedByVertice.clear();
    verticesInfluencedByLine.clear();
    neighborhoodLinesSet.clear();
    neighborhood.clear();

    init();
}

template <class BasicMapping>
void LineSetSkinningMapping<BasicMapping>::draw()
{
    //if (!getShow(this)) return;
    //glDisable (GL_LIGHTING);
    //glLineWidth(1);

    //glBegin (GL_LINES);

    //OutVecCoord& xto = *this->toModel->getX();
    //InVecCoord& xfrom = *this->fromModel->getX();
    //core::componentmodel::topology::Topology* topology = dynamic_cast<core::componentmodel::topology::Topology*>(this->fromModel->getContext()->getTopology());
//   topology::MeshTopology* t = dynamic_cast<topology::MeshTopology*>(topology);

    //for(unsigned int verticeIndex=0; verticeIndex<xto.size(); verticeIndex++)
    //{
    //	const topology::MeshTopology::Line& line = t->getLine(linesInfluencedByVertice[verticeIndex][0].lineIndex);
    //	Vector3 v = projectToSegment(xfrom[line[0]].getCenter(), xfrom[line[1]].getCenter(), xto[verticeIndex]);

    //	glColor3f (1,0,0);
    //	helper::gl::glVertexT(xto[verticeIndex]);
    //	helper::gl::glVertexT(v);

    //	for(unsigned int i=1; i<linesInfluencedByVertice[verticeIndex].size(); i++)
    //	{
    //		const topology::MeshTopology::Line& l = t->getLine(linesInfluencedByVertice[verticeIndex][i].lineIndex);
    //		Vector3 v = projectToSegment(xfrom[l[0]].getCenter(), xfrom[l[1]].getCenter(), xto[verticeIndex]);

    //		glColor3f (0,0,1);
    //		helper::gl::glVertexT(xto[verticeIndex]);
    //		helper::gl::glVertexT(v);
    //	}
    //}
    //glEnd();
}

template <class BasicMapping>
void LineSetSkinningMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    core::componentmodel::topology::Topology* topology = dynamic_cast<core::componentmodel::topology::Topology*>(this->fromModel->getContext()->getTopology());
    topology::MeshTopology* t = dynamic_cast<topology::MeshTopology*>(topology);

    for (unsigned int verticeIndex=0; verticeIndex<out.size(); verticeIndex++)
    {
        out[verticeIndex] = typename Out::Coord();
        for (unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
        {
            influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
            out[verticeIndex] += in[t->getLine(iline.lineIndex)[0]].getCenter()*iline.weight;
            out[verticeIndex] += in[t->getLine(iline.lineIndex)[0]].getOrientation().rotate(iline.position*iline.weight);
        }
    }
}

template <class BasicMapping>
void LineSetSkinningMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    core::componentmodel::topology::Topology* topology = dynamic_cast<core::componentmodel::topology::Topology*>(this->fromModel->getContext()->getTopology());
    topology::MeshTopology* t = dynamic_cast<topology::MeshTopology*>(topology);

    InVecCoord& xfrom = *this->fromModel->getX();


    for (unsigned int verticeIndex=0; verticeIndex<out.size(); verticeIndex++)
    {
        out[verticeIndex] = typename Out::Deriv();
        for (unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
        {
            influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
            Vector3 IP = xfrom[t->getLine(iline.lineIndex)[0]].getOrientation().rotate(iline.position);
            out[verticeIndex] += (in[t->getLine(iline.lineIndex)[0]].getVCenter() - IP.cross(in[t->getLine(iline.lineIndex)[0]].getVOrientation())) * iline.weight;
        }
    }
}


template <class BasicMapping>
void LineSetSkinningMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    core::componentmodel::topology::Topology* topology = dynamic_cast<core::componentmodel::topology::Topology*>(this->fromModel->getContext()->getTopology());
    topology::MeshTopology* t = dynamic_cast<topology::MeshTopology*>(topology);

    InVecCoord& xfrom = *this->fromModel->getX();

    //for(unsigned int lineIndex=0; lineIndex< (unsigned) t->getNbLines(); lineIndex++)
    //{
    //	for (unsigned int verticeInfluencedIndex=0; verticeInfluencedIndex<verticesInfluencedByLine[lineIndex].size(); verticeInfluencedIndex++)
    //	{
    //		influencedVerticeType vertice = verticesInfluencedByLine[lineIndex][verticeInfluencedIndex];
    //		Vector3 IP = xfrom[t->getLine(lineIndex)[0]].getOrientation().rotate(vertice.position);
    //		out[vertice.verticeIndex].getVCenter() += in[vertice.verticeIndex] * vertice.weight;
    //		out[vertice.verticeIndex].getVOrientation() += IP.cross(in[vertice.verticeIndex]) * vertice.weight;
    //	}
    //}
}

template <class BasicMapping>
void LineSetSkinningMapping<BasicMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    out.clear();
    out.resize(in.size());

    core::componentmodel::topology::Topology* topology = dynamic_cast<core::componentmodel::topology::Topology*>(this->fromModel->getContext()->getTopology());
    topology::MeshTopology* t = dynamic_cast<topology::MeshTopology*>(topology);

    InVecCoord& xfrom = *this->fromModel->getX();

    for(unsigned int i=0; i<in.size(); i++)
    {
        for (unsigned int j=0; j<in[i].size(); j++)
        {
            const OutSparseDeriv cIn = in[i][j];
            int verticeIndex = cIn.index;
            const OutDeriv d = (OutDeriv) cIn.data;
            //printf(" normale : %f %f %f",d.x(), d.y(), d.z());
            for(unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
            {
                influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
                Vector3 IP = xfrom[t->getLine(iline.lineIndex)[0]].getOrientation().rotate(iline.position);
                InDeriv direction;
                direction.getVCenter() = d * iline.weight;
                //printf("\n Weighted normale : %f %f %f",direction.getVCenter().x(), direction.getVCenter().y(), direction.getVCenter().z());
                direction.getVOrientation() = IP.cross(d) * iline.weight;
                out[i].push_back(InSparseDeriv(t->getLine(iline.lineIndex)[0], direction));
            }
        }
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
