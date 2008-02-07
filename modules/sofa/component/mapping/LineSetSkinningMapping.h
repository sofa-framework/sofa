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

#ifndef SOFA_COMPONENT_MAPPING_LINESETSKINNINGMAPPING_H
#define SOFA_COMPONENT_MAPPING_LINESETSKINNINGMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

template <class BasicMapping>
class LineSetSkinningMapping : public BasicMapping, public virtual core::objectmodel::BaseObject
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::SparseVecDeriv OutSparseVecDeriv;
    typedef typename Out::SparseDeriv OutSparseDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::SparseVecDeriv InSparseVecDeriv;
    typedef typename In::SparseDeriv InSparseDeriv;
    typedef typename In::Real Real;
    typedef typename OutCoord::value_type OutReal;


    LineSetSkinningMapping(In* from, Out* to)
        : Inherit(from, to)
        , nvNeighborhood(initData(&nvNeighborhood,(unsigned int)3,"neighborhoodLevel","Set the neighborhood line level"))
        , numberInfluencedLines(initData(&numberInfluencedLines,(unsigned int)4,"numberInfluencedLines","Set the number of most influenced lines by each vertice"))
        , weightCoef(initData(&weightCoef, (int) 4,"weightCoef","Set the coefficient used to compute the weight of lines"))
    {
    }

    virtual ~LineSetSkinningMapping()
    {
    }

    void init();

    void reinit();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );

    void draw();

protected:

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

    bool getShow(const core::objectmodel::BaseObject* m) const { return m->getContext()->getShowMappings(); }

    bool getShow(const core::componentmodel::behavior::BaseMechanicalMapping* m) const { return m->getContext()->getShowMechanicalMappings(); }

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
    Vector3 projectToSegment(Vector3& first, Vector3& last, OutCoord& vertice);

    /*!
    	Compute the weight betwewen a vertice and a line
    */
    double convolutionSegment(Vector3& first, Vector3& last, OutCoord& vertice);

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
