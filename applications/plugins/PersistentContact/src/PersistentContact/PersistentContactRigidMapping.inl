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
#ifndef SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTRIGIDMAPPING_INL
#define SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTRIGIDMAPPING_INL

#include "PersistentContactRigidMapping.h"

#include <sofa/component/mapping/nonlinear/RigidMapping.inl>

#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


template <class TIn, class TOut>
PersistentContactRigidMapping<TIn, TOut>::PersistentContactRigidMapping()
    : Inherit()
{
    m_inputMapping = 0;
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::beginAddContactPoint()
{
    if (!m_init)
    {
        if (this->f_printLog.getValue())
        {
            std::cout << "BeginAddContactPoint : pos = " << this->toModel->read(core::vec_id::read_access::position)->getValue() << " before suppr the contact" << std::endl;
        }

        m_previousPoints = this->d_points.getValue();
        this->clear(0);
        this->toModel->resize(0);
        m_init = true;
    }
}


template <class TIn, class TOut>
int PersistentContactRigidMapping<TIn, TOut>::addContactPointFromInputMapping(const sofa::type::Vec3& pos, std::vector< std::pair<int, double> > & /*baryCoords*/)
{
//	std::cout << "PersistentContactRigidMapping::addContactPointFromInputMapping()\n";

    if (this->f_printLog.getValue())
    {
        std::cout << "addContactPointFromInputMapping  Pos Ref = " << pos <<std::endl;
    }

    const typename In::VecCoord& xfrom = this->fromModel->read(core::vec_id::read_access::position)->getValue();

    Coord posContact;
    for (unsigned int i = 0; i < 3; i++)
        posContact[i] = (Real) pos[i];

    unsigned int inputIdx = m_inputMapping->d_index.getValue();

    this->d_index.setValue(inputIdx);
    this->d_rigidIndexPerPoint.setValue(m_inputMapping->d_rigidIndexPerPoint.getValue());
    Coord x_local = xfrom[inputIdx].inverseRotate(posContact - xfrom[inputIdx].getCenter());

    this->addPoint(x_local, inputIdx);

    int index = this->d_points.getValue().size() -1;
    this->toModel->resize(index+1);

    return index;
}


template <class TIn, class TOut>
int PersistentContactRigidMapping<TIn, TOut>::keepContactPointFromInputMapping(const int _index)
{
    if (this->f_printLog.getValue())
    {
        std::cout << "keepContactPointFromInputMapping index = " << _index <<std::endl;
    }

    unsigned int inputIdx = m_inputMapping->d_index.getValue();

    this->d_index.setValue(inputIdx);
    this->d_rigidIndexPerPoint.setValue(m_inputMapping->d_rigidIndexPerPoint.getValue());

    if (_index > (int)m_previousPoints.size())
    {
        std::cout << "\nKeepContactPointFromInputMapping Critical Error!!!!!\n";
        return 0;
    }

    this->addPoint(m_previousPoints[_index], inputIdx);

    int index = this->d_points.getValue().size() -1;
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
void PersistentContactRigidMapping<TIn, TOut>::bwdInit()
{
    const std::string path = this->m_nameOfInputMap.getValue();

    simulation::Node* parentNode = 0;
    parentNode = static_cast< simulation::Node* >(this->fromModel->getContext());

    if (parentNode)
    {
        type::vector< Inherit* > inherits;

        parentNode->getTreeObjects< Inherit, type::vector< Inherit* > >(&inherits);

        typename type::vector< Inherit* >::const_iterator it = inherits.begin();
        typename type::vector< Inherit* >::const_iterator itEnd = inherits.end();

        while (it != itEnd)
        {
            if ((*it)->getName() == path)
            {
                m_inputMapping = *it;
                break;
            }

            ++it;
        }
    }

    if (!m_inputMapping)
        msg_warning() << "Can not found the input mapping";
    else
        msg_info() << "Input mapping named " << m_inputMapping->getName() << " is found";
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::reset()
{
    setDefaultValues();
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::setDefaultValues()
{
    m_previousPosition = this->fromModel->read(core::vec_id::read_access::position)->getValue();
    m_previousFreePosition = this->fromModel->read(core::vec_id::read_access::position)->getValue();
    m_previousDx.resize(m_previousFreePosition.size());
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::handleEvent(sofa::core::objectmodel::Event* ev)
{
    if (dynamic_cast< simulation::AnimateEndEvent* >(ev))
    {
        storeFreePositionAndDx();
        m_init = false;
    }
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::storeFreePositionAndDx()
{
    m_previousFreePosition = this->fromModel->read(core::vec_id::read_access::freePosition)->getValue();
    m_previousDx = this->fromModel->read(core::vec_id::read_access::dx)->getValue();

    if (this->f_printLog.getValue())
    {
        std::cout << "===== end of the time ste =========\n stored Free Pos : " << m_previousFreePosition << std::endl;
        std::cout << " stored DX : " << m_previousDx << std::endl;
        std::cout << "============================" << std::endl;
    }

//    this->applyLinearizedPosition();

}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::applyLinearizedPosition()
{
    Data< VecCoord > newXFree;
    Data< InVecCoord > prevXFree;
    prevXFree.setValue(m_previousFreePosition);

    this->apply(0, newXFree, prevXFree);

    // We need to apply the previous position to obtain the right linearization
    Data< VecCoord > tempValue;
    Data< InVecCoord > prevX;
    prevX.setValue(m_previousPosition);
    this->apply(0, tempValue, prevX);

    Data< VecDeriv > newDx;
    Data< InVecDeriv > prevDx;
    prevDx.setValue(m_previousDx);

    this->applyJ(0, newDx, prevDx);

    Data< VecCoord >* newPos_d = this->toModel->write(core::vec_id::write_access::position);
    VecCoord &newPos = *newPos_d->beginEdit();

    newPos = newXFree.getValue();

    for (unsigned int i=0; i < newPos.size(); i++)
    {
        newPos[i] += newDx.getValue()[i];
    }

    newPos_d->endEdit();
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::applyPositionAndFreePosition()
{
    applyLinearizedPosition();
    core::Mapping<TIn, TOut>::apply(0, sofa::core::vec_id::write_access::freePosition, sofa::core::vec_id::read_access::freePosition);
    core::Mapping<TIn, TOut>::applyJ(0, sofa::core::vec_id::write_access::velocity, sofa::core::vec_id::read_access::velocity);
    core::Mapping<TIn, TOut>::applyJ(0, sofa::core::vec_id::write_access::freeVelocity, sofa::core::vec_id::read_access::freeVelocity);
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTRIGIDMAPPING_INL
