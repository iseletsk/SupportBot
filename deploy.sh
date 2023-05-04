
#scp -r *.py requirements.txt workdir 192.168.240.69:SupportBot

ssh -x 192.168.240.69 "cd SupportBot; pip install --upgrade -r requirements.txt"
ssh -x 192.168.240.69 "cd SupportBot; killall main.py"
ssh -x 192.168.240.69 "cd SupportBot; nohup python main.py >support.log 2>&1 &"