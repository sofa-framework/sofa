
/**************************************************************************************************************
*          If there is no context we have no means to obtain a processor value to set the task's Site         *
**************************************************************************************************************/
#ifdef NO_CONTEXT //Used to avoid calling getContext on classe without a getContext method. Ex: DataShared and DataPtrShared
    /* KAAPI_NUMBER_PARAMS is the number of possible parameters */
    template<class TASK,M4_PARAM(`class E$1', `', `,')>
  void Task(M4_PARAM(`const E$1& e$1', `', `,') )
   {

#ifdef RUN_TASK
 TASK()( M4_PARAM(`e$1', `', `,'));
#else
 a1::Fork<TASK>()( M4_PARAM(`e$1', `', `,'));
#endif  //end of RUN_TASK
if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
a1::Sync();

}

/**************************************************************************************************************
*                                   Code Used when there is a Context                                         *
**************************************************************************************************************/

#else
    /* KAAPI_NUMBER_PARAMS is the number of possible parameters */
    template<class TASK,M4_PARAM(`class E$1', `', `,')>
 void Task(M4_PARAM(`const E$1& e$1', `', `,') )
    {
	 RFO::Closure *c;
    
  Iterative::IterativePartition*p=prepareTask();
	if(p){
#ifdef SOFA_SMP_NUMA
		if(p->getCPU()!=-1)
			numa_set_preferred(p->getCPU()/2);
#endif
			
#ifdef RUN_TASK
 TASK()( M4_PARAM(`e$1', `', `,'));
#else
    c=     Iterative::Fork<TASK>(Iterative::SetPartition(*p))( M4_PARAM(`e$1', `', `,'));
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



	}else{


#ifdef RUN_TASK
 TASK()( M4_PARAM(`e$1', `', `,'));
 #else
	   c= 	Iterative::Fork<TASK>()( M4_PARAM(`e$1', `', `,'));
 
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
    /* KAAPI_NUMBER_PARAMS is the number of possible parameters */
    template<class TASK,class TASK2, M4_PARAM(`class E$1', `', `,')>
  void Task(M4_PARAM(`const E$1& e$1', `', `,') )
   {

#ifdef RUN_TASK
 TASK()( M4_PARAM(`e$1', `', `,'));
#else
 Iterative::Fork<TASK,TASK2>(DefaultAttribut())( M4_PARAM(`e$1', `', `,'));
#endif  //end of RUN_TASK
if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
a1::Sync();

}

/**************************************************************************************************************
*                                   Code Used when there is a Context                                         *
**************************************************************************************************************/

#else
    /* KAAPI_NUMBER_PARAMS is the number of possible parameters */
    template<class TASK,class TASK2,M4_PARAM(`class E$1', `', `,')>
 void Task(M4_PARAM(`const E$1& e$1', `', `,') )
    {
    
	 RFO::Closure *c;
  Iterative::IterativePartition*p=prepareTask();

	if(p){
#ifdef SOFA_SMP_NUMA
		if(p->getCPU()!=-1)
			numa_set_preferred(p->getCPU()/2);
#endif
			
#ifdef RUN_TASK
 TASK()( M4_PARAM(`e$1', `', `,'));
#else
    c=     Iterative::Fork<TASK,TASK2>(Iterative::SetPartition(*p))( M4_PARAM(`e$1', `', `,'));
//	INSERT_CLOSURE(this,c);

#endif //end of RUN_TASK



	}else{


#ifdef RUN_TASK
 TASK()( M4_PARAM(`e$1', `', `,'));
 #else
	   c= 	Iterative::Fork<TASK,TASK2> ( DefaultAttribut())( M4_PARAM(`e$1', `', `,'));
 
//	INSERT_CLOSURE(this,c);
#endif  //end of RUN_TASK


	}
if(sofa::core::CallContext::getExecutionType()!=sofa::core::CallContext::GRAPH_KAAPI)
a1::Sync();
    }
#endif
