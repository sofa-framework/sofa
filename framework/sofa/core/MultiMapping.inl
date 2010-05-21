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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_MULTIMAPPING_INL
#define SOFA_CORE_MULTIMAPPING_INL

#include <sofa/core/MultiMapping.h>

namespace sofa
{

namespace core
{



template < class In, class Out >
void MultiMapping<In,Out>::addInputModel(In* from)
{
    this->fromModels.push_back(from);
}

template< class In, class Out >
void MultiMapping<In,Out>::addOutputModel(Out* to)
{
    this->toModels.push_back(to);
}

template< class In, class Out>
helper::vector<In*>&  MultiMapping<In,Out>::getFromModels()
{
    return this->fromModels;
}

template< class In, class Out>
helper::vector<Out*>& MultiMapping<In,Out>::getToModels()
{
    return this->toModels;
}
template< class In, class Out >
helper::vector<objectmodel::BaseObject*> MultiMapping<In,Out>::getFrom()
{
    helper::vector<objectmodel::BaseObject*> base_fromModels;
    std::copy(fromModels.begin(), fromModels.end(), std::back_inserter(base_fromModels));
    return base_fromModels;
}

template< class In, class Out >
helper::vector<objectmodel::BaseObject* > MultiMapping<In,Out>::getTo()
{
    helper::vector<objectmodel::BaseObject*> base_toModels;
    std::copy(toModels.begin(), toModels.end(), std::back_inserter(base_toModels));
    return base_toModels;
}
template <class In, class Out>
void MultiMapping<In,Out>::init()
{
    this->updateMapping();
}


template <class In, class Out>
void MultiMapping<In,Out>::updateMapping()
{
    if( this->fromModels.empty() || this->toModels.empty() )
    {
        return;
    }

    const VecId &idCoord = VecId::position();
    helper::vector<OutVecCoord*> vecOutPos;
    getVecOutCoord(idCoord, vecOutPos);
    helper::vector<const InVecCoord*> vecInPos;
    getConstVecInCoord(idCoord, vecInPos);
    apply ( vecOutPos, vecInPos);


    const VecId &idDeriv = VecId::velocity();
    helper::vector<OutVecDeriv*> vecOutVel;
    getVecOutDeriv(idDeriv, vecOutVel);
    helper::vector<const InVecDeriv*> vecInVel;
    getConstVecInDeriv(idDeriv, vecInVel);
    applyJ( vecOutVel, vecInVel);
}


template <class In, class Out>
void MultiMapping<In,Out>::getVecInCoord(const VecId &id, helper::vector<InVecCoord*> &v) const
{
    for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(fromModels[i]->getVecCoord(id.index));
}

template <class In, class Out>
void MultiMapping<In,Out>::getVecInDeriv(const VecId &id, helper::vector<InVecDeriv*> &v) const
{
    for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(fromModels[i]->getVecDeriv(id.index));
}


template <class In, class Out>
void MultiMapping<In,Out>::getConstVecInDeriv(const VecId &id, helper::vector<const InVecDeriv*> &v) const
{
    for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(fromModels[i]->getVecDeriv(id.index));
}


template <class In, class Out>
void MultiMapping<In,Out>::getVecOutDeriv(const VecId &id, helper::vector<OutVecDeriv*> &v) const
{
    for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecDeriv(id.index));
}

template <class In, class Out>
void MultiMapping<In,Out>::getConstVecOutCoord(const VecId &id, helper::vector<const OutVecCoord*> &v) const
{
    for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecCoord(id.index));
}

template <class In, class Out>
void MultiMapping<In,Out>::getConstVecOutDeriv(const VecId &id, helper::vector<const OutVecDeriv*> &v) const
{
    for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecDeriv(id.index));
}



template <class In, class Out>
std::string MultiMapping<In,Out>::templateName(const MultiMapping<In, Out>* /*mapping*/)
{
    return std::string("MultiMapping<")+In::DataTypes::Name() + std::string(",") + Out::DataTypes::Name() + std::string(">");
}

template <class In, class Out>
void MultiMapping<In,Out>::disable()
{
}


} //core
} //sofa

#endif
