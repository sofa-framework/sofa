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
#include <sofa/component/mapping/linear/LineSetSkinningMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
type::Vec<3,double> LineSetSkinningMapping<TIn, TOut>::projectToSegment(const type::Vec<3,Real>& first, const type::Vec<3,Real>& last, const OutCoord& vertice)
{
    type::Vec3d segment,v_f,v_l;

    segment = last - first;
    v_f = vertice-first;
    v_l = vertice-last;

    if(v_f*segment>0.0 && -segment*v_l>0.0)
    {
        const double prod = v_f*segment;
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
double LineSetSkinningMapping<TIn, TOut>::convolutionSegment(const type::Vec<3,Real>& first, const type::Vec<3,Real>& last, const OutCoord& vertice)
{
    int steps = 1000;
    double sum = 0.0;
    type::Vec<3,Real> dist, line;

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
    const OutVecCoord& xto = this->toModel->read(core::ConstVecCoordId::position())->getValue();
    const InVecCoord& xfrom = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    m_topology = this->fromModel->getContext()->getMeshTopology();
    linesInfluencedByVertice.resize(xto.size());

    verticesInfluencedByLine.resize(m_topology->getNbLines());

    neighborhoodLinesSet.resize(m_topology->getNbLines());
    neighborhood.resize(m_topology->getNbLines());

    for(unsigned int line1Index=0; line1Index< (unsigned) m_topology->getNbLines(); line1Index++)
    {
        const sofa::core::topology::BaseMeshTopology::Line& line1 = m_topology->getLine(line1Index);
        for(unsigned int line2Index=0; line2Index< (unsigned) m_topology->getNbLines(); line2Index++)
        {
            const sofa::core::topology::BaseMeshTopology::Line& line2 = m_topology->getLine(line2Index);
            if ((line1[0] == line2[0]) || (line1[0] == line2[1]) || (line1[1] == line2[0]))
            {
                neighborhoodLinesSet[line1Index].insert(line2Index);
            }
        }
    }

    for(unsigned int line1Index=0; line1Index< (unsigned) m_topology->getNbLines(); line1Index++)
    {
        std::set<int> result;
        const std::insert_iterator<std::set<int> > res_ins(result, result.begin());

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

    if (m_topology->getNbLines() == 0)
        return;

    for(unsigned int verticeIndex=0; verticeIndex<xto.size(); verticeIndex++)
    {
        double	sumWeights = 0.0;
        type::vector<influencedLineType> lines;
        lines.resize(m_topology->getNbLines());

        for(unsigned int lineIndex=0; lineIndex< (unsigned) m_topology->getNbLines(); lineIndex++)
        {
            const sofa::core::topology::BaseMeshTopology::Line& line = m_topology->getLine(lineIndex);
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
        const int max = lines[lineInfluencedIndex].lineIndex;
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
void LineSetSkinningMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    const OutVecCoord& xto = this->toModel->read(core::ConstVecCoordId::position())->getValue();
    const InVecCoord& xfrom = this->fromModel->read(core::ConstVecCoordId::position())->getValue();

    std::vector<sofa::type::RGBAColor> colorVector;
    std::vector<sofa::type::Vec3> vertices;

    for (unsigned int verticeIndex=0; verticeIndex<xto.size(); verticeIndex++)
    {
        for (unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
        {

            influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
            const sofa::core::topology::BaseMeshTopology::Line& l = m_topology->getLine(linesInfluencedByVertice[verticeIndex][lineInfluencedIndex].lineIndex);
            type::Vec<3,Real> v = projectToSegment(xfrom[l[0]].getCenter(), xfrom[l[1]].getCenter(), xto[verticeIndex]);


            colorVector.push_back(sofa::type::RGBAColor(iline.weight, 0.0, (1.0-iline.weight),1.0));
            vertices.push_back(sofa::type::Vec3( xto[verticeIndex] ));
            vertices.push_back(sofa::type::Vec3( v ));
        }
    }
    vparams->drawTool()->drawLines(vertices,1,colorVector);
}

template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::apply( const sofa::core::MechanicalParams* mparams, OutDataVecCoord& outData, const InDataVecCoord& inData)
{
    SOFA_UNUSED(mparams);

    OutVecCoord& out = *outData.beginEdit();
    const InVecCoord& in = inData.getValue();

    for (unsigned int verticeIndex=0; verticeIndex<out.size(); verticeIndex++)
    {
        out[verticeIndex] = typename Out::Coord();
        for (unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
        {
            influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
            out[verticeIndex] += in[m_topology->getLine(iline.lineIndex)[0]].getCenter()*iline.weight;
            out[verticeIndex] += in[m_topology->getLine(iline.lineIndex)[0]].getOrientation().rotate(iline.position*iline.weight);
        }
    }
    outData.endEdit();
}

template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::applyJ( const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& outData, const InDataVecDeriv& inData)
{
    SOFA_UNUSED(mparams);

    const InVecCoord& xfrom = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    OutVecDeriv& out = *outData.beginEdit();
    const InVecDeriv& in = inData.getValue();
    for (unsigned int verticeIndex=0; verticeIndex<out.size(); verticeIndex++)
    {
        out[verticeIndex] = typename Out::Deriv();
        for (unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
        {
            influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
            type::Vec<3,Real> IP = xfrom[m_topology->getLine(iline.lineIndex)[0]].getOrientation().rotate(iline.position);
            out[verticeIndex] += (getVCenter(in[m_topology->getLine(iline.lineIndex)[0]]) - IP.cross(getVOrientation(in[m_topology->getLine(iline.lineIndex)[0]]))) * iline.weight;
        }
    }
    outData.endEdit();
}


template <class TIn, class TOut>
void LineSetSkinningMapping<TIn, TOut>::applyJT( const sofa::core::MechanicalParams* mparams, InDataVecDeriv& outData, const OutDataVecDeriv& inData)
{
    SOFA_UNUSED(mparams);

    InVecDeriv& out = *outData.beginEdit();
    const OutVecDeriv& in = inData.getValue();
    const InVecCoord& xfrom = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    out.clear();
    out.resize(xfrom.size());

    for (unsigned int verticeIndex=0; verticeIndex<in.size(); verticeIndex++)
    {
        typename Out::Deriv f = in[verticeIndex];

        for (unsigned int lineInfluencedIndex=0; lineInfluencedIndex<linesInfluencedByVertice[verticeIndex].size(); lineInfluencedIndex++)
        {
            influencedLineType iline = linesInfluencedByVertice[verticeIndex][lineInfluencedIndex];
            unsigned int I =m_topology->getLine(iline.lineIndex)[0];

            type::Vec<3,Real> IP = xfrom[I].getOrientation().rotate(iline.position);

            getVCenter(out[I]) += f * iline.weight;
            getVOrientation(out[I]) += IP.cross(f) *  iline.weight;

        }
    }

    outData.endEdit();

    /*
    	for(unsigned int lineIndex=0; lineIndex< (unsigned) m_topology->getNbLines(); lineIndex++)
    	{
    		unsigned int I = m_topology->getLine(lineIndex)[0];
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
void LineSetSkinningMapping<TIn, TOut>::applyJT( const sofa::core::ConstraintParams* mparams, InDataMatrixDeriv& outData, const OutDataMatrixDeriv& inData)
{
    SOFA_UNUSED(mparams);

    InMatrixDeriv& out = *outData.beginEdit();
    const OutMatrixDeriv& in = inData.getValue();
    const InVecCoord& xfrom = this->fromModel->read(core::ConstVecCoordId::position())->getValue();

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
                    type::Vec<3,Real> IP = xfrom[m_topology->getLine(iline.lineIndex)[0]].getOrientation().rotate(iline.position);
                    InDeriv direction;
                    getVCenter(direction) = data * iline.weight;
                    //printf("\n Weighted normale : %f %f %f",direction.getVCenter().x(), direction.getVCenter().y(), direction.getVCenter().z());
                    getVOrientation(direction) = IP.cross(data) * iline.weight;

                    o.addCol(m_topology->getLine(iline.lineIndex)[0], direction);
                }

                ++colIt;
            }
        }
    }

    outData.endEdit();
}

} //namespace sofa::component::mapping::linear
