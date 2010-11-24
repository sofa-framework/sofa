/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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

#ifndef SOFA_COMPONENT_MAPPING_LINESETSKINNINGMAPPING_INL
#define SOFA_COMPONENT_MAPPING_LINESETSKINNINGMAPPING_INL

#include <sofa/component/mapping/LineSetSkinningMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
Vec<3,double> LineSetSkinningMapping<TIn, TOut>::projectToSegment(const Vec<3,Real>& first, const Vec<3,Real>& last, const OutCoord& vertice)
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

template <class TIn, class TOut>
double LineSetSkinningMapping<TIn, TOut>::convolutionSegment(const Vec<3,Real>& first, const Vec<3,Real>& last, const OutCoord& vertice)
{
    int steps = 1000;
    double sum = 0.0;
    Vec<3,Real> dist, line;

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

template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::init()
{
    const OutVecCoord& xto = *this->toModel->getX();
    const InVecCoord& xfrom = *this->fromModel->getX();
    t = this->fromModel->getContext()->getMeshTopology();
    linesInfluencedByVertice.resize(xto.size());

    verticesInfluencedByLine.resize(t->getNbLines());

    neighborhoodLinesSet.resize(t->getNbLines());
    neighborhood.resize(t->getNbLines());

    for(unsigned int line1Index=0; line1Index< (unsigned) t->getNbLines(); line1Index++)
    {
        const sofa::core::topology::BaseMeshTopology::Line& line1 = t->getLine(line1Index);
        for(unsigned int line2Index=0; line2Index< (unsigned) t->getNbLines(); line2Index++)
        {
            const sofa::core::topology::BaseMeshTopology::Line& line2 = t->getLine(line2Index);
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
            const sofa::core::topology::BaseMeshTopology::Line& line = t->getLine(lineIndex);
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

template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::reinit()
{
    linesInfluencedByVertice.clear();
    verticesInfluencedByLine.clear();
    neighborhoodLinesSet.clear();
    neighborhood.clear();

    init();
}

template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::draw()
{
    if (!this->getShow()) return;
    glDisable (GL_LIGHTING);
    glLineWidth(1);

    glBegin (GL_LINES);

    const OutVecCoord& xto = *this->toModel->getX();
    const InVecCoord& xfrom = *this->fromModel->getX();

    for (unsigned int verticeIndex=0; verticeIndex<xto.size(); verticeIndex++)
    {
        //out[verticeIndex] = typename Out::Coord();
        for (unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
        {

            influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
            //Vec<3,Real> v = xfrom[t->getLine(iline.lineIndex)[0]].getCenter() + xfrom[t->getLine(iline.lineIndex)[0]].getOrientation().rotate(iline.position);
            const sofa::core::topology::BaseMeshTopology::Line& l = t->getLine(linesInfluencedByVertice[verticeIndex][lineInfluencedIndex].lineIndex);
            Vec<3,Real> v = projectToSegment(xfrom[l[0]].getCenter(), xfrom[l[1]].getCenter(), xto[verticeIndex]);


            glColor3f ((GLfloat) iline.weight, (GLfloat) 0, (GLfloat) (1.0-iline.weight));
            helper::gl::glVertexT(xto[verticeIndex]);
            helper::gl::glVertexT(v);

        }
    }

    glEnd();
}

template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
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

template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    const InVecCoord& xfrom = *this->fromModel->getX();

    for (unsigned int verticeIndex=0; verticeIndex<out.size(); verticeIndex++)
    {
        out[verticeIndex] = typename Out::Deriv();
        for (unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
        {
            influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
            Vec<3,Real> IP = xfrom[t->getLine(iline.lineIndex)[0]].getOrientation().rotate(iline.position);
            out[verticeIndex] += (getVCenter(in[t->getLine(iline.lineIndex)[0]]) - IP.cross(getVOrientation(in[t->getLine(iline.lineIndex)[0]]))) * iline.weight;
        }
    }
}


template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const InVecCoord& xfrom = *this->fromModel->getX();
    out.clear();
    out.resize(xfrom.size());

    for (unsigned int verticeIndex=0; verticeIndex<in.size(); verticeIndex++)
    {
        typename Out::Deriv f = in[verticeIndex];

        for (unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
        {
            influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
            unsigned int I =t->getLine(iline.lineIndex)[0];

            Vec<3,Real> IP = xfrom[I].getOrientation().rotate(iline.position);

            getVCenter(out[I]) += f * iline.weight;
            getVOrientation(out[I]) += IP.cross(f) *  iline.weight;

        }
    }


    /*
    	for(unsigned int lineIndex=0; lineIndex< (unsigned) t->getNbLines(); lineIndex++)
    	{
    		unsigned int I = t->getLine(lineIndex)[0];
    		for (unsigned int verticeInfluencedIndex=0; verticeInfluencedIndex<verticesInfluencedByLine[lineIndex].size(); verticeInfluencedIndex++)
    		{
    			influencedVerticeType vertice = verticesInfluencedByLine[lineIndex][verticeInfluencedIndex];
    			Vec<3,Real> IP = xfrom[I].getOrientation().rotate(vertice.position);
    			out[I].getVCenter() += in[vertice.verticeIndex] * vertice.weight;
    			out[I].getVOrientation() += IP.cross(in[vertice.verticeIndex]) * vertice.weight;
    		}
    	}
    */
}


template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    const InVecCoord& xfrom = *this->fromModel->getX();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            while (colIt != colItEnd)
            {
                const OutDeriv data = colIt.val();
                const unsigned int verticeIndex = colIt.index();

                //printf(" normale : %f %f %f",d.x(), d.y(), d.z());
                for (unsigned int lineInfluencedIndex = 0; lineInfluencedIndex < linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
                {
                    influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
                    Vec<3,Real> IP = xfrom[t->getLine(iline.lineIndex)[0]].getOrientation().rotate(iline.position);
                    InDeriv direction;
                    getVCenter(direction) = data * iline.weight;
                    //printf("\n Weighted normale : %f %f %f",direction.getVCenter().x(), direction.getVCenter().y(), direction.getVCenter().z());
                    getVOrientation(direction) = IP.cross(data) * iline.weight;

                    o.addCol(t->getLine(iline.lineIndex)[0], direction);
                }

                ++colIt;
            }
        }
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
