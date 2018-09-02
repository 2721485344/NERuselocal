#-*- coding: UTF-8 -*-

import sys
import os
# print(sys.argv)  # 返回一个list,第一个元素是程序本身路径
# print(sys.argv[0]) #E:/pythonWork3.6.2/深度学习/test.py  linux 返回 test.py
# print(sys.argv[1:])#[]
# print(sys.version)  # 返回  3.6.2 (v3.6.2:5fd33b5, Jul  8 2017, 04:57:36) [MSC v.1900 64 bit (AMD64)]
# print(sys.maxsize)  # 最大值 9223372036854775807
# print(sys.path)  # 返回模块的搜索路径
# # ['E:\\pythonWork3.6.2\\深度学习', 'E:\\pythonWork3.6.2', 'E:\\pythonWork3.6.2\\venv\\Scripts\\python36.zip', 'D:\\Python3.6.2\\DLLs', 'D:\\Python3.6.2\\lib', 'D:\\Python3.6.2', 'E:\\pythonWork3.6.2\\venv', 'E:\\pythonWork3.6.2\\venv\\lib\\site-packages']
# print(sys.platform)  # 返回操作系统平台名称win32
# sys.stdout.write("hello")
# val = sys.stdin.readline()[:-1]
# print("val",val)

print(os.getcwd()+os.sep)
import  tensorflow as tf
print(tf.app.flags.D)