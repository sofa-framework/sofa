/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v16.08                  *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
