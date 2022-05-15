# WasteClassification

start python script over ssh, without termination after closing: https://stackoverflow.com/questions/2975624/how-to-run-a-script-in-the-background-even-after-i-logout-ssh

start script: nohup ~/Code/WasteClassification/cam.py &

see script running: ps ax | grep cam.py

see all python scripts: ps -fA | grep python

kill: kill PID

ssh session display image on server:
export DISPLAY=:0

automatic login:
https://linuxconfig.org/how-to-enable-automatic-login-on-ubuntu-18-04-bionic-beaver-linux
