Tips and tricks
===============

Connecting to a remote swimpy brower interface
----------------------------------------------
The swimpy browser interface is using a local development server. If swimpy is
running on a remote (e.g. the cluster) you can still access it locally like this:

.. code-block:: console

    # log into remote with port forwarding
    $ ssh -v -L 8000:localhost:8000 <username>@cluster.pik-potsdam.de
    # start the swimpy browser on the remote
    $ swimpy browser start

Then navigate to `http://localhost:8000 <http://localhost:8000>`_ in your local browser.
