#-*-encoding=utf8-*-
import time
import requests
t=time.time()

r=requests.post('http://127.0.0.1:5002/?inputStr="乙肝和冠心病那个严重"')
print(r.text)