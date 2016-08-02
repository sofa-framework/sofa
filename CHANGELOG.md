
2014-06-??

- Added BaseData::::setRequired(), that marks a Data as not optional
- Added a MechanicalParams parameter to the Mapping::checkApply*() methods
- Massive code cleanup:
    - Removed all using directives ('using namespace') from all header files
    - Fix issues detected by cppcheck

2014-06-??

- Removed lots of deprecated code:
    - Removed the State::get*() methods (Note: use State::read(...) instead)
    - Removed the old API methods, that don't use MechanicalParams, namely:
        - ForceField::add*Force(...)
        - ForceField::getPotentialEnergy(...)
        - Mapping::apply*(...)
        - Mapping::computeAccFromMapping(...)
        - ProjectiveConstraintSets::project*(...)
        - Mass::accFromF(...)
        - Mass::addMDx(...)
        - Mass::getKineticEnergy(...)
        - Mass::getPotentialEnergy(...)
        - Mass::addBToMatrix(...)
        - Mass::addKToMatrix(...)
    - Removed the following deprecated methods:
        - Base::findField(const std::string &name) const
        - Base::addField(BaseData* f, const char* name)
        - VisualModel::initTextures()
        - VisualModel::update()
        - BaseLink::ConvertOldPath(...)
        - CollisionModel::getGroup()
        - CollisionModel::setGroup(const int groupId)
        - BaseObject::draw()
        - MeshTopology::isCubeActive(int index)
        - Data::setHelpMsg(const char* val)

2014-06-24

- Moved modules files around: they are now organised by module rather than by
  namespace.
- Added scripts/fix-modules-includes.sh to fix the includes in existing
  projects.
