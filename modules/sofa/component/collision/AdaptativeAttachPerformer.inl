#ifndef SOFA_COMPONENT_COLLISION_ADAPTATIVEATTACHPERFORMER_INL
#define SOFA_COMPONENT_COLLISION_ADAPTATIVEATTACHPERFORMER_INL

#include <sofa/component/collision/AdaptativeAttachPerformer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/MouseInteractor.h>

namespace sofa
{

namespace component
{

namespace collision
{

template <class DataTypes>
AdaptativeAttachPerformer<DataTypes>::AdaptativeAttachPerformer(BaseMouseInteractor *i):
    TInteractionPerformer<DataTypes>(i),
    mapper(NULL)
{
    flags.setShowVisualModels(false);
    flags.setShowInteractionForceFields(true);
}

template <class DataTypes>
AdaptativeAttachPerformer<DataTypes>::~AdaptativeAttachPerformer()
{
    clear();
}

template <class DataTypes>
bool AdaptativeAttachPerformer<DataTypes>::start_partial(const BodyPicked& picked)
{
    core::behavior::MechanicalState<DataTypes>* mstateCollision=NULL;
    if (picked.body)
    {
        mapper = MouseContactMapper::Create(picked.body);
        if (!mapper)
        {
            this->interactor->serr << "Problem with Mouse Mapper creation : " << this->interactor->sendl;
            return false;
        }
        std::string name = "contactMouse";
        mstateCollision = mapper->createMapping(name.c_str());
        mapper->resize(1);

        const typename DataTypes::Coord pointPicked=picked.point;
        const int idx=picked.indexCollisionElement;
        typename DataTypes::Real r=0.0;

        this->index = mapper->addPointB(pointPicked, idx, r
                          #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                  , picked.baryCoords
                          #endif
                                  );
        mapper->update();

        if (mstateCollision->getContext() != picked.body->getContext())
        {

            simulation::Node *mappedNode=(simulation::Node *) mstateCollision->getContext();
            simulation::Node *mainNode=(simulation::Node *) picked.body->getContext();
            core::behavior::BaseMechanicalState *mainDof=dynamic_cast<core::behavior::BaseMechanicalState *>(mainNode->getMechanicalState());
            const core::objectmodel::TagSet &tags=mainDof->getTags();
            for (core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
            {
                mstateCollision->addTag(*it);
                mappedNode->mechanicalMapping->addTag(*it);
            }
            mstateCollision->setName("AttachedPoint");
            mappedNode->mechanicalMapping->setName("MouseMapping");
        }
    }
    else
    {
        mstateCollision = dynamic_cast< core::behavior::MechanicalState<DataTypes>*  >(picked.mstate);
        this->index = picked.indexCollisionElement;
        if (!mstateCollision)
        {
            this->interactor->serr << "incompatible MState during Mouse Interaction " << this->interactor->sendl;
            return false;
        }
    }

    using sofa::component::interactionforcefield::ARPSStiffSpringForceField;

    m_forcefield = sofa::core::objectmodel::New< ARPSStiffSpringForceField<DataTypes> >(dynamic_cast<MouseContainer*>(this->interactor->getMouseContainer()), mstateCollision);
    ARPSStiffSpringForceField< DataTypes >* arpsstiffspringforcefield = static_cast< ARPSStiffSpringForceField< DataTypes >* >(m_forcefield.get());
    arpsstiffspringforcefield->setName("Spring-Mouse-Contact");
    arpsstiffspringforcefield->setArrowSize((float)this->size);
    arpsstiffspringforcefield->setDrawMode(2); //Arrow mode if size > 0


    arpsstiffspringforcefield->addSpring(0,this->index, stiffness, 0.0, picked.dist);
    const core::objectmodel::TagSet &tags=mstateCollision->getTags();
    for (core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
        arpsstiffspringforcefield->addTag(*it);

    mstateCollision->getContext()->addObject(arpsstiffspringforcefield);

    return true;
}

template <class DataTypes>
void AdaptativeAttachPerformer<DataTypes>::start()
{
    if (m_forcefield)
    {
        clear();
        return;
    }
    BodyPicked picked=this->interactor->getBodyPicked();
    if (!picked.body && !picked.mstate) return;

    if (!start_partial(picked)) return; //template specialized code is here

    double distanceFromMouse=picked.rayLength;
    this->interactor->setDistanceFromMouse(distanceFromMouse);
    Ray ray = this->interactor->getMouseRayModel()->getRay(0);
    ray.setOrigin(ray.origin() + ray.direction()*distanceFromMouse);
    sofa::core::BaseMapping *mapping;
    this->interactor->getContext()->get(mapping); assert(mapping);
    mapping->apply(core::MechanicalParams::defaultInstance());
    mapping->applyJ(core::MechanicalParams::defaultInstance());
    m_forcefield->init();
    this->interactor->setMouseAttached(true);
}



template <class DataTypes>
void AdaptativeAttachPerformer<DataTypes>::execute()
{
    sofa::core::BaseMapping *mapping;
    this->interactor->getContext()->get(mapping); assert(mapping);
    mapping->apply(core::MechanicalParams::defaultInstance());
    mapping->applyJ(core::MechanicalParams::defaultInstance());
    this->interactor->setMouseAttached(true);
}

template <class DataTypes>
void AdaptativeAttachPerformer<DataTypes>::clear()
{
    if (m_forcefield)
    {
        using sofa::component::interactionforcefield::ARPSStiffSpringForceField;
        ARPSStiffSpringForceField< DataTypes >* arpsstiffspringforcefield = static_cast< ARPSStiffSpringForceField< DataTypes >* >(m_forcefield.get());
        arpsstiffspringforcefield->removeSpringForce(this->index, arpsstiffspringforcefield->previousForce[0]);

        m_forcefield->cleanup();
        m_forcefield->getContext()->removeObject(m_forcefield);
        m_forcefield.reset();
    }

    if (mapper)
    {
        mapper->cleanup();
        delete mapper; mapper=NULL;
    }

    this->interactor->setDistanceFromMouse(0);
    this->interactor->setMouseAttached(false);
}

template <class DataTypes>
void AdaptativeAttachPerformer<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (m_forcefield)
    {
        core::visual::VisualParams* vp = const_cast<core::visual::VisualParams*>(vparams);
        core::visual::DisplayFlags backup = vp->displayFlags();
        vp->displayFlags() = flags;
        m_forcefield->draw(vp);
        vp->displayFlags() = backup;
    }
}
}
}
}
#endif
