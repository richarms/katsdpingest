Thread pools
------------
The actual sending and receiving of packets is done by separate C threads.
Each stream is associated with a *thread pool*, which is a pool of threads
able to process its packets. For small numbers of streams, one can use a thread
pool per stream. With large numbers of streams (more than the number of CPUs),
it may be better to use a single thread pool with a thread per CPU core.

There is one important consideration for deciding whether streams share a
thread pool: if a received stream is not being consumed, it may block one of
the threads from the thread pool [#]_. Thus, if several streams share a thread
pool, it is important to be responsive to all of them. Deciding that one
stream is temporarily uninteresting and can be discarded while listening only
to another one can thus lead to a deadlock if the two streams share a thread
pool with only one thread.

.. [#] This is a limitation of the current design that will hopefully be
   overcome in future versions.

.. py:currentmodule:: spead2

.. py:class:: spead2.ThreadPool(threads=1)

   Construct a thread pool and start the threads.

   .. py:method:: stop()

      Shut down the worker threads. Calling this while there are still open
      streams is not advised. In most cases, garbage collection is sufficient.
