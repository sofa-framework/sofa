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

#ifndef SOFA_COMPONENT_MAPPING_LINESETSKINNINGMAPPING_H
#define SOFA_COMPONENT_MAPPING_LINESETSKINNINGMAPPING_H

#include <sofa/core/Mapping.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/Vec3Types.h>

#include <sofa/SofaCommon.h>

#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
class LineSetSkinningMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(LineSetSkinningMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename OutCoord::value_type OutReal;

    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::Real Real;

    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

protected:
    LineSetSkinningMapping()
        : Inherit()
        , nvNeighborhood(initData(&nvNeighborhood,(unsigned int)3,"neighborhoodLevel","Set the neighborhood line level"))
        , numberInfluencedLines(initData(&numberInfluencedLines,(unsigned int)4,"numberInfluencedLines","Set the number of most influenced lines by each vertice"))
        , weightCoef(initData(&weightCoef, (int) 4,"weightCoef","Set the coefficient used to compute the weight of lines"))
    {
    }

    virtual ~LineSetSkinningMapping()
    {
    }
public:
    void init();

    void reinit();

    virtual void apply( const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecCoord& out, const InDataVecCoord& in);
    //void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    virtual void applyJ( const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecDeriv& out, const InDataVecDeriv& in);
    //void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    virtual void applyJT( const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, InDataVecDeriv& out, const OutDataVecDeriv& in);
    //void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    virtual void applyJT( const sofa::core::ConstraintParams* mparams /* PARAMS FIRST */, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in);
    //void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );

    void draw(const core::visual::VisualParams* vparams);

protected:

    sofa::core::topology::BaseMeshTopology* t;

    /*!
    	Set the neighborhood line level
    */
    Data<unsigned int> nvNeighborhood;

    /*!
    	Set the number of most influenced lines by each vertice
    */
    Data<unsigned int> numberInfluencedLines;

    /*!
    	Set the coefficient used to compute the weight of lines
    */
    Data<int> weightCoef;

private:

    /*!
    	Class to store the index, local weight, and local position of a line
    */
    class influencedLineType
    {
    public:
        influencedLineType()
        {
            weight = 0.0;
        };
        ~influencedLineType() {};
        int lineIndex;
        double weight;
        OutCoord position;
    };

    /*!
    	Class to store the index, local weight, and local position of a vertice
    */
    class influencedVerticeType
    {
    public:
        influencedVerticeType()
        {
            weight = 0.0;
        };
        ~influencedVerticeType() {};
        int verticeIndex;
        double weight;
        OutCoord position;
    };

    /*!
    	Compute the perpendicular distance from a vertice to a line
    */
    defaulttype::Vec<3,double> projectToSegment(const defaulttype::Vec<3,Real>& first, const defaulttype::Vec<3,Real>& last, const OutCoord& vertice);

    /*!
    	Compute the weight betwewen a vertice and a line
    */
    double convolutionSegment(const defaulttype::Vec<3,Real>& first, const defaulttype::Vec<3,Real>& last, const OutCoord& vertice);

    /*!
    	Stores the lines influenced by each vertice
    */
    vector<vector<influencedLineType> > linesInfluencedByVertice;

    /*!
    	Stores the vertices influenced by each line
    */
    vector<vector<influencedVerticeType> > verticesInfluencedByLine;

    /*!
    	Stores the first level line neighborhood
    */
    vector<std::set<int> > neighborhoodLinesSet;

    /*!
    	Stores the n level line neighborhood
    */
    vector<std::set<int> > neighborhood;
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
