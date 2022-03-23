Usage
=====
This page contains :ref:`simple examples <simple_examples>` of
pricing path-dependent options. For detailed documentation, refer to
:doc:`api`.


.. _installation:

Installation
------------

To obtain the latest version of ``qdpmc``, use pip:

.. code-block:: bash

    pip install qdpmc

.. _simple_examples:

Get started
-----------

Import modules
^^^^^^^^^^^^^^^^^

.. ipython:: python

    import qdpmc as qm
    import numpy as np

Set up parameters
^^^^^^^^^^^^^^^^^
Before pricing products, specify parameters of Monte Carlo
simulation and market dynamics.

.. ipython:: python

    # Simulation parameters
    mc = qm.MonteCarlo(batch_size=125, num_iter=1000)

    # Black-Scholes dynamics
    bs = qm.BlackScholes(r=0.03, q=0, v=0.25, day_counter=252)

Single-barrier options
^^^^^^^^^^^^^^^^^^^^^^
This example in the index page demonstrates the
pricing of an up-and-out call option using ``qdpmc``.
To price an up-and-out put option, modify the associated codes:

.. ipython:: python

    # Specify an up-and-out call option
    up_out_call = qm.UpOut(
        spot=100,
        barrier=150,
        rebate=0,
        ob_days=np.linspace(1, 252, 252),
        payoff=qm.Payoff(
            qm.plain_vanilla,
            strike=100,
            option_type="call"
        )
    )
    # PV and Greek letters
    up_out_call.calc_value(mc, bs)

Calculate Greeks
^^^^^^^^^^^^^^^^
Request Greeks by specifying ``request_greeks`` in ``calc_value``:

.. ipython:: python

    up_out_call.calc_value(mc, bs, request_greeks=True)


By default, the method uses central differences for Delta, Rho, and Vega.
Change the steps and schemes by specifying the associated parameters.

.. ipython:: python

    up_out_call.calc_value(mc, bs, request_greeks=True,
                           fd_steps={'ds':0.01, 'dr': 0.01, 'dv': 0.01},
                           fd_scheme={'ds': 'central', 'dr': 'forward',
                                      'dv': 'backward'})

Re-use random numbers
^^^^^^^^^^^^^^^^^^^^^
By default, each time ``calc_value`` is run a different set of random
numbers is used (a different entropy is used to initialize
the random number generator.
Check out `this page <https://numpy.org/doc/stable/reference/random/
bit_generators/generated/numpy.random.SeedSequence.html>`__
for details about ``NumPy`` random number generator).
To reuse the same set of random numbers, run the following codes:

.. ipython:: python

    e = mc.most_recent_entropy
    up_out_call.calc_value(mc, bs, entropy=e)

Time-varying barrier level
^^^^^^^^^^^^^^^^^^^^^^^^^^
If a scalar is passed to ``barrier``, the barrier is then assumed to
be time-invariant. Pass an array to ``barrier`` to specify a time-varying
barrier level. However, note that this array should match the length of
``ob_days``.

.. ipython:: python

    time_varying_barrier = np.linspace(120, 130, 252)
    time_varying_barrier_option = qm.UpOut(
        spot=100,
        barrier=time_varying_barrier,
        rebate=0,
        ob_days=np.linspace(1, 252, 252),
        payoff=qm.Payoff(
            qm.plain_vanilla,
            strike=100,
            option_type="call"
        )
    )
    time_varying_barrier_option.calc_value(mc, bs)

Define payoff function
^^^^^^^^^^^^^^^^^^^^^^
To customize a payoff function, define a *Payoff*:

.. ipython:: python

    def dsf_payoff(s, k1, k2):
        call = qm.plain_vanilla(s, k1, 'call')
        put = qm.plain_vanilla(s, k2, 'put')
        return call + put

    dsf = qm.DoubleOut(
        spot=100,
        barrier_up=120,
        barrier_down=90,
        ob_days_up=np.linspace(1, 252, 252),
        ob_days_down=np.linspace(21, 252, 12),
        payoff=qm.Payoff(
            func=dsf_payoff,
            k1=100, k2=110
        ),
        rebate_up=3,
        rebate_down=2
    )

    dsf.calc_value(mc, bs)
