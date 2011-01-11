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
#ifndef SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTRIGIDMAPPING_INL
#define SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTRIGIDMAPPING_INL

#include "PersistentContactRigidMapping.h"

#include <sofa/component/mapping/RigidMapping.inl>

#include <sofa/simulation/common/AnimateEndEvent.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


template <class TIn, class TOut>
PersistentContactRigidMapping<TIn, TOut>::PersistentContactRigidMapping(core::State< In >* from, core::State< Out >* to)
    : Inherit(from, to)
    , contactDuplicate(initData(&contactDuplicate, false, "contactDuplicate", "if true, this mapping is a copy of an input mapping and is used to gather contact points (PersistentFrictionContact Response)"))
    , nameOfInputMap(initData(&nameOfInputMap, "nameOfInputMap", "if contactDuplicate==true, it provides the name of the input mapping"))
{
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::beginAddContactPoint()
{
    if (!m_init)
    {
        std::cout << "BeginAddContactPoint : pos = " << (*this->toModel->getX()) << " before suppr the contact" << std::endl;
        m_previousPoints = this->points.getValue();
        this->clear(0);
        this->toModel->resize(0);
        m_init = true;
    }
    else
    {
        m_init = false;
    }
}


template <class TIn, class TOut>
int PersistentContactRigidMapping<TIn, TOut>::addContactPointFromInputMapping(const sofa::defaulttype::Vector3& pos, std::vector< std::pair<int, double> > & /*baryCoords*/)
{
    std::cout << "addContactPointFromInputMapping  Pos Ref = " << pos <<std::endl;
    const typename In::VecCoord& xfrom = *this->fromModel->getX();

    Coord posContact;
    for (unsigned int i = 0; i < 3; i++)
        posContact[i] = (Real) pos[i];

    unsigned int inputIdx = m_inputMapping->index.getValue();

    this->index.setValue(inputIdx);
    this->repartition.setValue(m_inputMapping->repartition.getValue());
    Coord x_local = xfrom[inputIdx].inverseRotate(posContact - xfrom[inputIdx].getCenter());

    this->addPoint(x_local, inputIdx);

    int index = this->points.getValue().size() -1;
    this->toModel->resize(index+1);

    return index;
}


template <class TIn, class TOut>
int PersistentContactRigidMapping<TIn, TOut>::keepContactPointFromInputMapping(const int _index)
{
    std::cout << "keepContactPointFromInputMapping index = " << _index <<std::endl;

    unsigned int inputIdx = m_inputMapping->index.getValue();

    this->index.setValue(inputIdx);
    this->repartition.setValue(m_inputMapping->repartition.getValue());

    if (_index > (int)m_previousPoints.size())
    {
        std::cout << "KeepContactPointFromInputMapping Critical Error\n";
        return 0;
    }

    this->addPoint(m_previousPoints[_index], inputIdx);

    int index = this->points.getValue().size() -1;
    this->toModel->resize(index+1);

    return index;
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::init()
{
    this->f_listening.setValue(true);
    m_init = false;

    setDefaultValues();

    this->Inherit::init();
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::reset()
{
    setDefaultValues();
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::setDefaultValues()
{
    m_previousFreePosition = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    m_previousDx.resize(m_previousFreePosition.size());
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::bwdInit()
{
    if (contactDuplicate.getValue())
    {
        const std::string path = nameOfInputMap.getValue();
        this->fromModel->getContext()->get(m_inputMapping, sofa::core::objectmodel::BaseContext::SearchRoot);

        if (!m_inputMapping)
            serr << "WARNING : can not found the input mapping" << sendl;
        else
            sout << "Input mapping named " << m_inputMapping->getName() << " is found" << sendl;
    }
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::handleEvent(sofa::core::objectmodel::Event* ev)
{
    if (dynamic_cast< simulation::AnimateEndEvent* >(ev))
    {
        storeFreePositionAndDx();
    }
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::storeFreePositionAndDx()
{
    m_previousFreePosition = this->fromModel->read(core::ConstVecCoordId::freePosition())->getValue();
    m_previousDx = this->fromModel->read(core::ConstVecDerivId::dx())->getValue();
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::applyLinearizedPosition()
{
    Data< VecCoord > newXFree;
    Data< InVecCoord > prevXFree;
    prevXFree.setValue(m_previousFreePosition);

    this->apply(newXFree, prevXFree, 0);

    Data< VecDeriv > newDx;
    Data< InVecDeriv > prevDx;
    prevDx.setValue(m_previousDx);

    this->applyJ(newDx, prevDx, 0);

    Data< VecCoord >* newPos_d = this->toModel->write(core::VecCoordId::position());
    VecCoord &newPos = *newPos_d->beginEdit();

    newPos = newXFree.getValue();

    for (unsigned int i=0; i < newPos.size(); i++)
    {
        newPos[i] += newDx.getValue()[i];
    }

    newPos_d->endEdit();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTRIGIDMAPPING_INL
