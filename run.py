# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__ = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__ = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

import sys, os
from ldmtools import *
import imageio
import numpy as np
from neubiaswg5 import CLASS_LNDDET
from cytomine.models import *
import joblib
from neubiaswg5.helpers.data_upload import imwrite
from neubiaswg5.helpers import NeubiasJob, prepare_data, upload_data, upload_metrics, get_discipline
from VotingTreeRegressor import VotingTreeRegressor

def find_by_attribute(att_fil, attr, val):
	return next(iter([i for i in att_fil if hasattr(i, attr) and getattr(i, attr) == val]), None)

def	get_neubias_coords(gt_path, tr_im):
	first_im = imageio.imread(os.path.join(gt_path, '%d.tif' % tr_im[0]))
	nldms = np.max(first_im)
	nimages = len(tr_im)
	xcs = np.zeros((nimages, nldms))
	ycs = np.zeros((nimages, nldms))
	xrs = np.zeros((nimages, nldms))
	yrs = np.zeros((nimages, nldms))
	for i in range(len(tr_im)):
		id = tr_im[i]
		gt_img = imageio.imread(os.path.join(gt_path, '%d.tif'%id))
		for id_term in range(1, nldms+1):
			(y, x) = np.where(gt_img==id_term)
			(h, w) = gt_img.shape
			yc = y[0]
			xc = x[0]
			yr = yc/h
			xr = xc/w
			xcs[i, id_term-1] = xc
			ycs[i, id_term-1] = yc
			xrs[i, id_term-1] = xr
			yrs[i, id_term-1] = yr
	return np.array(xcs), np.array(ycs), np.array(xrs), np.array(yrs)


def build_vote_map(repository, image_number, clf, h2, v2, h3, v3, sq, stepc):
	intg = build_integral_image(readimage(repository, image_number, image_type='tif'))
	(h, w) = intg.shape

	vote_map = np.zeros((h, w))

	coords = np.array([[x, y] for x in range(0, w, stepc) for y in range(0, h, stepc)]).astype(int)

	y_v = coords[:, 1]
	x_v = coords[:, 0]

	step = 50000

	b = 0

	while (b < x_v.size):
		b_next = min(b + step, x_v.size)
		offsets = clf.predict(compute_features(intg, x_v[b:b_next], y_v[b:b_next], h2, v2, h3, v3, sq))
		n_trees = len(offsets)
		off_size = int(b_next - b)

		offsets = np.array(offsets)
		toffsize = off_size * n_trees
		offsets = offsets.reshape((toffsize, 2))

		offsets[:, 0] = np.tile(x_v[b:b_next], n_trees) - offsets[:, 0]
		offsets[:, 1] = np.tile(y_v[b:b_next], n_trees) - offsets[:, 1]

		t, = np.where(offsets[:, 0] < 0)
		offsets = np.delete(offsets, t, axis=0)
		t, = np.where(offsets[:, 1] < 0)
		offsets = np.delete(offsets, t, axis=0)
		t, = np.where(offsets[:, 0] >= w)
		offsets = np.delete(offsets, t, axis=0)
		t, = np.where(offsets[:, 1] >= h)
		offsets = np.delete(offsets, t, axis=0)
		(toffsize, tamere) = offsets.shape
		for i in range(toffsize):
			vote_map[int(offsets[i, 1]), int(offsets[i, 0])] += 1

		b = b_next

	return vote_map


def find_best_positions(vote_map, coords, R):
	(h, w, nldms) = vote_map.shape

	cs = np.zeros(2 * nldms)
	for ip in range(nldms):

		x_begin = int(min(w - 1, max(0, coords[ip] - R)))
		x_end = int(max(0, min(coords[ip] + R + 1, w - 1)))

		y_begin = int(min(h - 1, max(0, coords[ip + nldms] - R)))
		y_end = int(max(0, min(h - 1, coords[ip + nldms] + R + 1)))

		if (x_begin != x_end and y_begin != y_end):
			window = vote_map[y_begin:y_end, x_begin:x_end, ip]
			(y, x) = np.where(window == np.max(window))
			cs[ip] = x[0] + x_begin
			cs[ip + nldms] = y[0] + y_begin
		elif (x_begin == x_end and y_begin != y_end):
			window = vote_map[y_begin:y_end, x_begin, ip]
			y, = np.where(window == np.max(window))
			cs[ip] = x_begin
			cs[ip + nldms] = y[0] + y_begin
		elif (y_begin == y_end and x_begin != x_end):
			window = vote_map[y_begin, x_begin:x_end, ip]
			x, = np.where(window == np.max(window))
			cs[ip + nldms] = y_begin
			cs[ip] = x[0] + x_begin
		else:
			cs[ip] = x_begin
			cs[ip + nldms] = y_begin

	return cs


def fit_shape(mu, P, ty):
	y = np.copy(ty)

	(nldms, k) = P.shape
	b = np.zeros((k, 1))
	nldm = int(nldms / 2)
	c = np.zeros((2, nldm))
	new_y = np.zeros(nldms)

	m_1 = np.mean(y[:nldm])
	m_2 = np.mean(y[nldm:])

	y[:nldm] = y[:nldm] - m_1
	y[nldm:] = y[nldm:] - m_2

	ite = 0
	theta = 0
	s = 0
	while (ite < 100):
		x = mu + np.dot(P, b)
		n2 = np.linalg.norm(y) ** 2
		a = (np.dot(y, x) / n2)[0]
		b = np.sum((y[:nldm] * x[nldm:]) - (x[:nldm] * y[nldm:])) / n2
		s = np.sqrt((a ** 2) + (b ** 2))
		theta = np.arctan(b / a)
		scaling_matrix = s * np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
		c[0, :] = y[:nldm]
		c[1, :] = y[nldm:]
		new_c = np.dot(scaling_matrix, c)
		new_y[:nldm] = new_c[0, :]
		new_y[nldm:] = new_c[1, :]
		b = np.dot(P.T, new_y.reshape((nldms, 1)) - mu)
		ite += 1

	s = 1. / s
	theta = -theta
	scaling_matrix = s * np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
	c[0, :] = x[:nldm].reshape(nldm)
	c[1, :] = x[nldm:].reshape(nldm)
	new_c = np.dot(scaling_matrix, c)
	new_y[:nldm] = new_c[0, :] + m_1
	new_y[nldm:] = new_c[1, :] + m_2
	return new_y

def main():
	with NeubiasJob.from_cli(sys.argv) as conn:
		problem_cls = get_discipline(conn, default=CLASS_LNDDET)
		conn.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization of the prediction phase")
		in_images, gt_images, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, conn, is_2d=True, **conn.flags)
		list_imgs = [int(image.rstrip('.tif')) for image in os.listdir(in_path) if image.endswith('.tif')]
		train_job = Job().fetch(conn.parameters.model_to_use)
		properties = PropertyCollection(train_job).fetch()
		str_terms = ""
		for prop in properties:
			if prop.fetch(key='id_terms') != None:
				str_terms = prop.fetch(key='id_terms').value
		term_list = [int(x) for x in str_terms.split(' ')]
		attached_files = AttachedFileCollection(train_job).fetch()

		feature_file = find_by_attribute(attached_files, "filename", "features.joblib")
		feature_filepath = os.path.join(in_path, "features.joblib")
		feature_file.download(feature_filepath, override=True)
		(h2,v2,h3,v3,sq) = joblib.load(feature_filepath)

		coords_file = find_by_attribute(attached_files, "filename", "coords.joblib")
		coords_filepath = os.path.join(in_path, "coords.joblib")
		coords_file.download(coords_filepath, override=True)
		(Xc, Yc) = joblib.load(coords_filepath)
		(nims, nldms) = Xc.shape

		coords = np.zeros(2 * nldms)
		i = 0
		for id_term in conn.monitor(term_list, start=10, end=50, period = 0.05, prefix="Building vote maps..."):
			model_file = find_by_attribute(attached_files, "filename", "%d_model.joblib" % id_term)
			model_filepath = os.path.join(in_path, "%d_model.joblib" % id_term)
			model_file.download(model_filepath, override=True)
			clf = joblib.load(model_filepath)
			mx = np.mean(Xc[:, id_term-1])
			my = np.mean(Yc[:, id_term-1])
			coords[i] = mx
			coords[i+nldms] = my
			i+=1
			for j in list_imgs:
				print(j)
				vote_map = build_vote_map(in_path, j, clf, h2, v2, h3, v3, sq, conn.parameters.model_step)
				np.savez_compressed('%d_%d_votemap.npy' % (j, id_term), vote_map)

		muP_file = find_by_attribute(attached_files, "filename", "muP.joblib")
		muP_filepath = os.path.join(in_path, "muP.joblib")
		muP_file.download(muP_filepath, override=True)
		(mu, P) = joblib.load(muP_filepath)
		(nims, nldms) = Xc.shape
		for id_img in conn.monitor(list_imgs, start=50, end=80, period = 0.05, prefix="Post-processing..."):
			probability_map = np.load('%d_%d_votemap.npy.npz' % (id_img, term_list[0]))['arr_0']
			(hpmap,wpmap) = probability_map.shape
			probability_volume = np.zeros((hpmap,wpmap,len(term_list)))
			probability_volume[:,:,0] = probability_map
			for i in range(1,len(term_list)):
				id_term = term_list[i]
				probability_volume[:, :, i] = np.load('%d_%d_votemap.npy.npz'%(id_img, id_term))['arr_0']
			current_R = conn.parameters.model_R_MAX
			while current_R >= conn.parameters.model_R_MIN:
				coords = np.round(find_best_positions(probability_volume, coords, int(np.round(current_R)))).astype(int)
				coords = np.round(fit_shape(mu, P, coords)).astype(int)
				current_R = current_R * conn.parameters.model_alpha
			x_final = np.round(coords[:nldms])
			y_final = np.round(coords[nldms:])
			lbl_img = np.zeros((hpmap, wpmap), 'uint8')
			for i in range(nldms):
				lbl_img[int(y_final[i]), int(x_final[i])] = term_list[i]
			imwrite(path=os.path.join(out_path, '%d.tif' % id_img), image=lbl_img.astype(np.uint8), is_2d=True)

		upload_data(problem_cls, conn, in_images, out_path, **conn.flags, is_2d=True,
					monitor_params={"start": 80, "end": 90, "period": 0.1})
		conn.job.update(progress=90, statusComment="Computing and uploading metrics (if necessary)...")
		upload_metrics(problem_cls, conn, in_images, gt_path, out_path, tmp_path, **conn.flags)
		conn.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")

if __name__ == "__main__":
	main()