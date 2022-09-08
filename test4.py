# author:Hurricane
# date:  2022/9/3
# E-mail:hurri_cane@qq.com

# def mk(x):
# 	def mk1( ):
# 		print("Decorated")
# 		x()
# 	return mk1
# def mk2():
# 	print("ordinary")
# p= mk(mk2)
# p()
dic = dict.fromkeys( ["k1" , "k2" ,'k3'],[])
dic[ 'k1' ].append(1)
dic[ "k2" ].append(2)
dic[ 'k1']= 1
print(dic)
