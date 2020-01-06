

import csv
import random
import pickle
import argparse

#category = ['texture-related','fabric-related','shape-related','part-related','style-related']
category = ['clothes category','clothes button','clothes color','clothes length','clothes pattern','clothes shape','collar shape','sleeve length','sleeve shape']


category_num = {
	'clothes category':20,
	'clothes button':13,
	'clothes color':55,
	'clothes length':7,
	'clothes pattern':28,
	'clothes shape':11,
	'collar shape':26,
	'sleeve length':8,
	'sleeve shape':17,
	}

#triplet_num = 100000

#root1 = '/home/nesa320/DATA/fashionAI/train/Annotations'
root1 = './data/DARN'
#root1 = '.'


def make_dataset(triplet_num=100000,other=''):

	#filename_csv = root1 + '/label'+other+'.csv'
	filename = root1 + '/filenames_'+other+'.txt' #filenames_{train | valid | test}.txt
	reader = open(filename,'r')
	#reader = csv.reader(csvFile)
	data = []
	for item in reader:
		data.append(item)
	reader.close()

	label = root1 + '/'+other+'_label'+'.txt' #{train | valid | test}_label.txt
	reader_label = open(label,'r')
	#reader = csv.reader(csvFile)
	data_label = []
	for item in reader_label:
		data_label.append(item)
	reader_label.close()


	#File = open(root1+'/type_attr.txt','br')
	#type_attr = pickle.load(File)

	category_dict = {}

	triplets = []
	for c in category:
		category_dict[c] = []

	for item_i in range(len(data)):
		item = data_label[item_i]
		attr_num = int((len(item))/2)

		for i in range(attr_num):
			category_dict[ category[int(item[i*2+1])] ].append([data[item_i],int(item[i*2+2])])
		#if item[2].find('m') == -1:
		#	category_dict[item[1]].append([item[0], item[2].find('y')])

	#print(len(data))
	print('data generation')
	for i in range(triplet_num):
		if i % int(triplet_num/4) == 0:
			print(100 * i / triplet_num,'%')

		#随机选取一个大类
		cate_r = random.randint(0, len(category)-1)

		
		cate_sub = random.randint(0, category_num[category[cate_r]]-1)
		#cate_sub = type_attr[cate_r][cate_sub]

		#print(cate_sub)
		#选取一个满足小类的图片a
		while True:
			a = random.randint(0, len(category_dict[category[cate_r]])-1)
			if category_dict[category[cate_r]][a][1] == cate_sub:
				break
			#else:
			#	print(category_dict[category[cate_r]][a][1]-cate_sub)

		#选取一个不满足小类的图片b
		while True:
			b = random.randint(0, len(category_dict[category[cate_r]])-1)
			if  category_dict[category[cate_r]][b][1] != cate_sub:
				break
			
		#选取一个满足小类且不等于a的图片c
		while True:
			c = random.randint(0, len(category_dict[category[cate_r]])-1)
			#if a != c and category_dict[category[cate_r]][c][1] == cate_sub:
			if category_dict[category[cate_r]][c][1] == cate_sub:
				break
			
		#将 [[a,a_label],[b,b_label],[c,c_label],category] 加入triplets中
		triplets.append([category_dict[category[cate_r]][a],category_dict[category[cate_r]][b],category_dict[category[cate_r]][c],cate_r])

	return triplets
if __name__ == '__main__':

	triplets = make_dataset()

	File = open('./triplets.txt','wb')
	pickle.dump(triplets, File)
	File.close()
