
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/CallContext.h>
#include<IterativePartition.h>

namespace sofa
{

namespace core
{
namespace objectmodel
{
  class BaseObjectTasks{
 	public:
   virtual const BaseContext* getContext() const=0;
   virtual BaseContext* getContext()=0;
   virtual Iterative::IterativePartition*  getPartition()=0;
   virtual Iterative::IterativePartition*  prepareTask()=0;
