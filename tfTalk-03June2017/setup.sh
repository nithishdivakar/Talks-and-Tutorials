svn checkout https://github.com/nithishdivakar/Talks-and-Tutorials/trunk/tfTalk-03June2017                                                                     
cd tfTalk-03June2017
# virtualenv env --no-site-packages
virtualenv env --no-site-packages --python=$(which python2)
source env/bin/activate
pip install -r requirements.txt
mkdir -p 05cnn/dataset/cifar
curl -L -o 05cnn/dataset/cifar/cifar-10-python.tar.gz http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
deactivate
