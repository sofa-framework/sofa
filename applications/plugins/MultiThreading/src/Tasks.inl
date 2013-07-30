#ifndef MultiThreadingTasks_inl__
#define MultiThreadingTasks_inl__

namespace sofa
{

	namespace simulation
	{



		inline Task::Status::Status()
			: mBusy(0L)
		{
		}

		inline bool Task::Status::IsBusy() const
		{
			return mBusy.operator long() != 0L;
		}

		inline void Task::Status::MarkBusy(bool bBusy)
		{
			if (bBusy)
			{
				++mBusy;
			}
			else
			{
				--mBusy;
			}
		}



		inline Task::Status* Task::getStatus(void) const 
		{
			return const_cast<Task::Status*>(m_Status);
		}


	} // namespace simulation

} // namespace sofa


#endif