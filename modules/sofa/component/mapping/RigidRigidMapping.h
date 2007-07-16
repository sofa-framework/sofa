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
#ifndef SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_H

#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/VisualModel.h>
#include <vector>

using namespace sofa::defaulttype;

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
class RigidRigidMapping : public BasicMapping, public core::VisualModel
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::SparseDeriv InSparseDeriv;
    typedef typename Coord::value_type Real;
    enum { N=Coord::static_size };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef Vec<N,Real> Vec;

protected:
    std::vector<Coord> points;
    std::vector<Coord> pointsR0;
    Mat rotation;
    class Loader;
    void load(const char* filename);
    DataField<sofa::helper::vector<unsigned int> >  repartition;

public:
    DataField<unsigned> index;

    RigidRigidMapping(In* from, Out* to)
        : Inherit(from, to)
        , index(dataField(&index,(unsigned)0,"index","input DOF index"))
        , repartition(dataField(&repartition,"repartition","number of dest dofs per entry dof"))
    {
    }

    virtual ~RigidRigidMapping()
    {
    }

    void init();

    void parse(core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("filename"))
            this->load(arg->getAttribute("filename"));
        this->Inherit::parse(arg);
    }

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    //void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

    void clear();

    void setRepartition(unsigned int value);
    void setRepartition(std::vector<unsigned int> values);

protected:

    bool getShow(const core::objectmodel::BaseObject* m) const { return m->getContext()->getShowMappings(); }

    bool getShow(const core::componentmodel::behavior::BaseMechanicalMapping* m) const { return m->getContext()->getShowMechanicalMappings(); }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
