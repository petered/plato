
from IPython.lib import passwd
import os
hashed_password = passwd()
profile_name = 'nbserver'

os.system('ipython profile create %s' % profile_name)

config_file_contents = """
# Following directions in: http://ipython.org/ipython-doc/1/interactive/public_server.html
c = get_config()

# Kernel config
c.IPKernelApp.pylab = 'inline'  # if you want plotting support always

# Notebook config
#c.NotebookApp.certfile = u'/absolute/path/to/your/certificate/mycert.pem'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
# It is a good idea to put it on a known, fixed port
c.NotebookApp.port = 9999
c.NotebookApp.password = u'%s'
""" % (hashed_password, )

config_file_loc = os.path.join(os.getenv('HOME'), '.ipython/profile_%s/ipython_notebook_config.py' % (profile_name, ))

with open(config_file_loc, 'w') as f:
    f.write(config_file_contents)

print 'Created Profile %s' % profile_name
