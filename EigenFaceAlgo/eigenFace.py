# Implementation of EigenFace Algorithm on Yale dataset
# the dataset has been modified by their name
# 7 random faces of each person is choosen for training 
# Remaining 4 faces of each person is taken for testing 
# the original dimension of the image is taken into account

import cv2, glob
import sys
import numpy as np
import os
from numpy import array
from sklearn.decomposition import PCA
import statistics 
import pandas
import random
import shutil

#from the choosen data set 
face_count = 15

train_face_count = 7
test_face_count = 4

l = train_face_count * face_count

m = 320
n = 243

mn = m * n

face_dir = 'Yale_dataset/'
# creating a global matrix for storing all selected image as column in the flaoting array
A = np.empty(shape=(mn, l), dtype='float64')
# sorting the indices of all selected images
train_ids = []

eig_vals = []
face_ids = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
def eigen_model():
	print('>>>--Finding the EigenFaces.......')
	cur_img = 0
	# reading the images and calcuting global matrix of selected image dataset
	for face_id in face_ids:
    	# choosing randomly any two images from set of a three images
		ids = random.sample(range(1, 12), train_face_count)
		train_ids.append(ids)
        # reading two images from the set of three
		for train_id in ids:
                # getting the path of the image choosen above
				if(train_id <= 9):
					img_path = os.path.join(face_dir + str(face_id) + str(train_id) + '.png')
				else:
					f_id = int(face_id/10)
					img_path = os.path.join(face_dir + str(f_id) + str(train_id) + '.png')
				# reading the image from the path
				img = cv2.imread(img_path, 0)
                # converting image into floating point column array
				# print(img)
				img_col =  np.array(img, dtype='float64').flatten()
				# forming the matrix for the entire image selected 
				# print(img_col)
				A[:, cur_img] = img_col[:]
				cur_img += 1
	
	# calculating mean for every image
	mean_img_col = np.sum(A, axis=1) / l
	# print(mean_img_col)
	#subtracting mean from the images
	for i in range(0, l):
		A[:, i] -= mean_img_col[i]
	
	# calculating the convariance matrix for the entire sets of images selected
	C = np.matrix(A.transpose()) * np.matrix(A)
	C /= l
	# calculating the eigen values and vectors from the convariance matrix
	eig_vals, eig_vecs = np.linalg.eig(C)
	# sorting eigen values and vectors in descending order
	sort_indices = eig_vals.argsort()[::-1]
	eig_vals = eig_vals[sort_indices]
	eig_vecs = eig_vecs[:,sort_indices]
	
	# selecting values of k as 50 (for now..)
	k = 100
	eig_vals = eig_vals[0:k]
	eig_vecs = eig_vecs[:, 0:k]
	# getting left eigen vector
	eig_vecs = A * eig_vecs
	# finding norm of each vector
	norms = np.linalg.norm(eig_vecs, axis=0)
	# normalizing the eigen vectors
	eig_vecs /= norms

	print('>>>--EigenFaces found!\n')

	# finding the weight vector as 
	W = eig_vecs.transpose() * A

	return(W, A, mean_img_col, eig_vals, eig_vecs)

def evaluating(W, A, mean_img_col, eig_vals, eig_vecs):
	# checking the results directory for is existance 
	if not os.path.exists('Y_res'):
   		os.makedirs('Y_res')
	else:
		shutil.rmtree('Y_res')
		os.makedirs('Y_res')
	# calling the eigen model function
	print('>>>--Evaluating The DataSet Choosen.......')
	# print(mean_img_col, W)
	# creating the result file for more information
	results_file = os.path.join('Y_res', 'results.txt')
	# opening the file..
	f = open(results_file, 'w')
	# calcuting t
	test_count = test_face_count * face_count
	test_correct = 0
	i = 0
	img_count = 0
	for face_id in face_ids:
		for test_id in range(1, 12):
			# checking if the image has been selected for traininig or not
			if((test_id in train_ids[i]) == False):
				img_count += 1
				# getting the path to the image..
				if(test_id <= 9):
					img_path = os.path.join(face_dir + str(face_id) + str(test_id) + '.png')
				else:
					f_id = int(face_id/10)
					img_path = os.path.join(face_dir + str(f_id) + str(test_id) + '.png')
				#reading the image from the path 
				img = cv2.imread(img_path, 0)
				# converting image into a row vector 
				img_col =  np.array(img, dtype='float64').flatten()
				# subtracting the mean from the image vector
				img_col -= mean_img_col
				# converting image row vector to column vector
				img_col = np.reshape(img_col, (mn, 1))
				# projecting image vector to the eigenspace
				S = eig_vecs.transpose() * img_col
				# subtracting the projected vector from the weight vector
				diff = W - S
				# finding the norms of the difference calculated above
				norms = np.linalg.norm(diff, axis = 0)
				# taking minimum ||W_j - S||
				closest_face_id = np.argmin(norms) 
				# getting the face number of the sample image
				result_id = 100 * ((closest_face_id / train_face_count) + 1)
				# checking for the face number is present in the train dataset
				result = (result_id == face_id)
				if result == True:
					test_correct += 1
					f.write('image: %s\nresult: correct\n\n' % img_path)
				else:
					f.write('image: %s\nresult: wrong, got %2d\n\n' %(img_path, result_id))
		i += 1
	print('>>>--DataSet Evaluation Done!')
	print('Total Correct--' ,test_correct)
	print('Total Incorrect--' ,(test_count-test_correct))
	accuracy = float(100. * test_correct / test_count)
	print ('Accuracy: ' + str(accuracy)) 
	f.write('Accuracy: %.2f\n' % (accuracy))
	f.close()                         

if __name__ == "__main__":
	W, A, mean_img_col, eig_vals, eig_vecs = eigen_model()
	evaluating(W, A, mean_img_col, eig_vals, eig_vecs)
