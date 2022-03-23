.. qdpmc documentation master file, created by
   sphinx-quickstart on Mon Jul 19 10:30:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. note::

   This package is under active development.

   Feedback and questions: dev@yieldchain.com

``qdpmc`` is a Python package for pricing path-dependent options
and structured products via Monte Carlo simulation. It utilizes
*vectorization* to boost algorithm speed. It offers a *simple*, *intuitive* ,
and *flexible* API to its users. This simple example demonstrates how it works:


.. ipython:: python

   import qdpmc as qm

   import numpy as np

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

    # Simulation parameters
    mc = qm.MonteCarlo(batch_size=125, num_iter=1000)

    # Black-Scholes dynamics
    bs = qm.BlackScholes(r=0.03, q=0, v=0.25, day_counter=252)

    # PV and Greek letters
    up_out_call.calc_value(mc, bs)

Checkout :doc:`usage` for further information.

Contents
========

.. toctree::

   usage
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
