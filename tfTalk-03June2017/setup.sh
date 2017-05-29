mkdir tfTalk-03June2017
cd tfTalk-03June2017
#virtualenv env --no-site-packages
virtualenv env --no-site-packages --python=/usr/bin/python2
. env/bin/activate
curl -OL https://raw.githubusercontent.com/nithishdivakar/Talks-and-Tutorials/master/tfTalk-03June2017/requirements.txt

pip install -r requirements.txt

deactivate
