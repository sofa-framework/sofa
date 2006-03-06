#include "FnDispatcher.inl"
#include "FnDispatcher.h"

namespace Sofa
{

namespace Components
{

namespace Common
{

template class FnDispatcher<Abstract::CollisionElement, bool>;
template class FnDispatcher<Abstract::CollisionElement, Collision::DetectionOutput*>;


} // namespace Common

} // namespace Components

} // namepsace Sofa
