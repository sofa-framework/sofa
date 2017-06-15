/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/CallContext.h>
#include<IterativePartition.h>

namespace sofa
{

namespace core
{
namespace objectmodel
{
class BaseObjectTasks
{
public:
    virtual const BaseContext* getContext() const=0;
    virtual BaseContext* getContext()=0;
    virtual Iterative::IterativePartition*  getPartition()=0;
    virtual Iterative::IterativePartition*  prepareTask()=0;

    virtual ~BaseObjectTasks();








    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 1 is the number of possible parameters */
    template<class TASK,class E1>
    void Task(const E1& e1 )
    {

#ifdef RUN_TASK
        TASK()( e1);
#else
        a1::Fork<TASK>()( e1);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 1 is the number of possible parameters */
    template<class TASK,class E1>
    void Task(const E1& e1 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1);
#else
            c= 	Iterative::Fork<TASK>()( e1);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 1 is the number of possible parameters */
    template<class TASK,class TASK2, class E1>
    void Task(const E1& e1 )
    {

#ifdef RUN_TASK
        TASK()( e1);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 1 is the number of possible parameters */
    template<class TASK,class TASK2,class E1>
    void Task(const E1& e1 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif









    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 2 is the number of possible parameters */
    template<class TASK,class E1,class E2>
    void Task(const E1& e1,const E2& e2 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2);
#else
        a1::Fork<TASK>()( e1,e2);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 2 is the number of possible parameters */
    template<class TASK,class E1,class E2>
    void Task(const E1& e1,const E2& e2 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1,e2);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2);
#else
            c= 	Iterative::Fork<TASK>()( e1,e2);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 2 is the number of possible parameters */
    template<class TASK,class TASK2, class E1,class E2>
    void Task(const E1& e1,const E2& e2 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1,e2);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 2 is the number of possible parameters */
    template<class TASK,class TASK2,class E1,class E2>
    void Task(const E1& e1,const E2& e2 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1,e2);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1,e2);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif









    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 3 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3>
    void Task(const E1& e1,const E2& e2,const E3& e3 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3);
#else
        a1::Fork<TASK>()( e1,e2,e3);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 3 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3>
    void Task(const E1& e1,const E2& e2,const E3& e3 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1,e2,e3);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3);
#else
            c= 	Iterative::Fork<TASK>()( e1,e2,e3);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 3 is the number of possible parameters */
    template<class TASK,class TASK2, class E1,class E2,class E3>
    void Task(const E1& e1,const E2& e2,const E3& e3 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1,e2,e3);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 3 is the number of possible parameters */
    template<class TASK,class TASK2,class E1,class E2,class E3>
    void Task(const E1& e1,const E2& e2,const E3& e3 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1,e2,e3);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1,e2,e3);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif









    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 4 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4);
#else
        a1::Fork<TASK>()( e1,e2,e3,e4);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 4 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1,e2,e3,e4);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4);
#else
            c= 	Iterative::Fork<TASK>()( e1,e2,e3,e4);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 4 is the number of possible parameters */
    template<class TASK,class TASK2, class E1,class E2,class E3,class E4>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1,e2,e3,e4);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 4 is the number of possible parameters */
    template<class TASK,class TASK2,class E1,class E2,class E3,class E4>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1,e2,e3,e4);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1,e2,e3,e4);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif









    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 5 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5);
#else
        a1::Fork<TASK>()( e1,e2,e3,e4,e5);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 5 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5);
#else
            c= 	Iterative::Fork<TASK>()( e1,e2,e3,e4,e5);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 5 is the number of possible parameters */
    template<class TASK,class TASK2, class E1,class E2,class E3,class E4,class E5>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1,e2,e3,e4,e5);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 5 is the number of possible parameters */
    template<class TASK,class TASK2,class E1,class E2,class E3,class E4,class E5>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1,e2,e3,e4,e5);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif









    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 6 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6);
#else
        a1::Fork<TASK>()( e1,e2,e3,e4,e5,e6);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 6 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6);
#else
            c= 	Iterative::Fork<TASK>()( e1,e2,e3,e4,e5,e6);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 6 is the number of possible parameters */
    template<class TASK,class TASK2, class E1,class E2,class E3,class E4,class E5,class E6>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1,e2,e3,e4,e5,e6);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 6 is the number of possible parameters */
    template<class TASK,class TASK2,class E1,class E2,class E3,class E4,class E5,class E6>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1,e2,e3,e4,e5,e6);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif









    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 7 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6,class E7>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6,e7);
#else
        a1::Fork<TASK>()( e1,e2,e3,e4,e5,e6,e7);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 7 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6,class E7>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6,e7);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7);
#else
            c= 	Iterative::Fork<TASK>()( e1,e2,e3,e4,e5,e6,e7);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 7 is the number of possible parameters */
    template<class TASK,class TASK2, class E1,class E2,class E3,class E4,class E5,class E6,class E7>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6,e7);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1,e2,e3,e4,e5,e6,e7);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 7 is the number of possible parameters */
    template<class TASK,class TASK2,class E1,class E2,class E3,class E4,class E5,class E6,class E7>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6,e7);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1,e2,e3,e4,e5,e6,e7);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif









    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 8 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6,e7,e8);
#else
        a1::Fork<TASK>()( e1,e2,e3,e4,e5,e6,e7,e8);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 8 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6,e7,e8);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8);
#else
            c= 	Iterative::Fork<TASK>()( e1,e2,e3,e4,e5,e6,e7,e8);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 8 is the number of possible parameters */
    template<class TASK,class TASK2, class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6,e7,e8);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1,e2,e3,e4,e5,e6,e7,e8);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 8 is the number of possible parameters */
    template<class TASK,class TASK2,class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6,e7,e8);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1,e2,e3,e4,e5,e6,e7,e8);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif









    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 9 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8,class E9>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8,const E9& e9 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9);
#else
        a1::Fork<TASK>()( e1,e2,e3,e4,e5,e6,e7,e8,e9);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 9 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8,class E9>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8,const E9& e9 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6,e7,e8,e9);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9);
#else
            c= 	Iterative::Fork<TASK>()( e1,e2,e3,e4,e5,e6,e7,e8,e9);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 9 is the number of possible parameters */
    template<class TASK,class TASK2, class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8,class E9>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8,const E9& e9 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1,e2,e3,e4,e5,e6,e7,e8,e9);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 9 is the number of possible parameters */
    template<class TASK,class TASK2,class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8,class E9>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8,const E9& e9 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6,e7,e8,e9);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1,e2,e3,e4,e5,e6,e7,e8,e9);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif









    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 10 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8,class E9,class E10>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8,const E9& e9,const E10& e10 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
#else
        a1::Fork<TASK>()( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 10 is the number of possible parameters */
    template<class TASK,class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8,class E9,class E10>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8,const E9& e9,const E10& e10 )
    {
        RFO::Closure *c;

        Iterative::IterativePartition*p=prepareTask();
        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
#else
            c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
#else
            c= 	Iterative::Fork<TASK>()( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
///
/// Tasks with two implementations
///



    /**************************************************************************************************************
    *          If there is no context we have no means to obtain a processor value to set the task's Site         *
    **************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* 10 is the number of possible parameters */
    template<class TASK,class TASK2, class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8,class E9,class E10>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8,const E9& e9,const E10& e10 )
    {

#ifdef RUN_TASK
        TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
#else
        Iterative::Fork<TASK,TASK2>(DefaultAttribut())( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
#endif  //end of RUN_TASK
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();

    }

    /**************************************************************************************************************
    *                                   Code Used when there is a Context                                         *
    **************************************************************************************************************/

#else
    /* 10 is the number of possible parameters */
    template<class TASK,class TASK2,class E1,class E2,class E3,class E4,class E5,class E6,class E7,class E8,class E9,class E10>
    void Task(const E1& e1,const E2& e2,const E3& e3,const E4& e4,const E5& e5,const E6& e6,const E7& e7,const E8& e8,const E9& e9,const E10& e10 )
    {

        RFO::Closure *c;
        Iterative::IterativePartition*p=prepareTask();

        if(p)
        {
#ifdef SOFA_SMP_NUMA
            if(p->getCPU()!=-1)
                numa_set_preferred(p->getCPU()/2);
#endif

#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
#else
            c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



        }
        else
        {


#ifdef RUN_TASK
            TASK()( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);
#else
            c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( e1,e2,e3,e4,e5,e6,e7,e8,e9,e10);

//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


        }
        if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
            a1::Sync();
    }
#endif
}; //class BaseObjectTasks
}// objectmodel
}// core
}// sofa
