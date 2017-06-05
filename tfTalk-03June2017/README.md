## Hi there. 

Welcome to the tutorial. 

The content in this folder is meant to accompany the slides of the tutorial. 
To download it please do the following steps. 

**Step 0** Make sure you have subversion installed on your system. Please type `svn --version` to verify if its there. If not, please check the section installing subversion at the end

**Step 1** In terminal, naviagate to a convinient location in your system.

**Step 2** Download the setup script using **one of the** following ways.
1. Directly download from this [link](https://raw.githubusercontent.com/nithishdivakar/Talks-and-Tutorials/master/tfTalk-03June2017/setup.sh)
2. Execute command `curl -OL https://raw.githubusercontent.com/nithishdivakar/Talks-and-Tutorials/master/tfTalk-03June2017/setup.sh`
3. Execute command `wget https://raw.githubusercontent.com/nithishdivakar/Talks-and-Tutorials/master/tfTalk-03June2017/setup.sh`


**Step 3** Run `sh setup.sh` in terminal.

*Step 3* downloads the contents of this folder to your system and sets up a python virtual environment. 
Then it installes all the necessary python packages like tensorflow into the virtual environment.
Finally it downloads some necessary (small) datasets into convinient/required locations. 
This step might take some time to finish if you are on slow internet connections.

### Installing subversion
ubuntu: `apt-get install subversion`

arch linux: `pacman -S subversion`

centos: `yum install subversion`
