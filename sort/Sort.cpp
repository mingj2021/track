#include "Sort.h"
#include "linear_assignment.h"
#include <iostream>

int KalmanBoxTracker::count = 1;

KalmanBoxTracker::KalmanBoxTracker(bbox_t bbox) :kf(8, 4, 0), meas(4,1,CV_32F)
{
	id = count++;
	initialize_kalman(bbox);
}

KalmanBoxTracker::~KalmanBoxTracker()
{

}


void KalmanBoxTracker::initialize_kalman(bbox_t bbox)
{
	//状态转移矩阵A
	std::vector<float> transition_vec = {
		1., 0., 0., 0., 1., 0., 0., 0.,
		0., 1., 0., 0., 0., 1., 0., 0.,
		0., 0., 1., 0., 0., 0., 1., 0.,
		0., 0., 0., 1., 0., 0., 0., 1.,
		0., 0., 0., 0., 1., 0., 0., 0.,
		0., 0., 0., 0., 0., 1., 0., 0.,
		0., 0., 0., 0., 0., 0., 1., 0.,
		0., 0., 0., 0., 0., 0., 0., 1. };
	cv::Mat transitionMatrix(8, 8, CV_32F, transition_vec.data());
	kf.transitionMatrix = transitionMatrix.clone();
	//std::cout << "transitionMatrix=" << kf.transitionMatrix << std::endl;

	//测量矩阵H
	std::vector<float> measurement_vec = {
		1., 0., 0., 0., 0., 0., 0., 0.,
		0., 1., 0., 0., 0., 0., 0., 0.,
		0., 0., 1., 0., 0., 0., 0., 0.,
		0., 0., 0., 1., 0., 0., 0., 0.
	};
	cv::Mat measurementMatrix(4, 8, CV_32F, measurement_vec.data());
	kf.measurementMatrix = measurementMatrix.clone() * 1.0;
	//std::cout << "measurementMatrix=" << kf.measurementMatrix << std::endl;

	//系统误差Q
	kf.processNoiseCov = cv::Mat::eye(8, 8, CV_32F)*1.0;
	kf.processNoiseCov(cv::Range(7, 8), cv::Range(7, 8)) *= 0.01;
	kf.processNoiseCov(cv::Range(4, 8), cv::Range(4, 8)) *= 0.01;
	//std::cout << "processNoiseCov=" << kf.processNoiseCov << std::endl;

	//测量误差R
	kf.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1.0;
	kf.measurementNoiseCov(cv::Range(2, 4), cv::Range(2, 4)) *= 10;
	//std::cout << "measurementNoiseCov=" << kf.measurementNoiseCov << std::endl;

	//最小均方误差P
	kf.errorCovPost = cv::Mat::eye(8, 8, CV_32F) * 1.0;
	kf.errorCovPost(cv::Range(4, 8), cv::Range(4, 8)) *= 1000;
	kf.errorCovPost *= 10;
	//std::cout << "errorCovPost=" << kf.errorCovPost << std::endl;

	//初始值
	kf.statePost = cv::Mat::zeros(8, 1, CV_32F);
	float centre_x = bbox.x + bbox.w / 2.;
	float centre_y = bbox.y + bbox.h / 2.;
	float a = 1.0 * bbox.w / bbox.h;
	meas.at<float>(0) = centre_x;
	meas.at<float>(1) = centre_y;
	meas.at<float>(2) = a;
	meas.at<float>(3) = bbox.h;
	kf.statePost.at<float>(0) = centre_x;
	kf.statePost.at<float>(1) = centre_y;
	kf.statePost.at<float>(2) = a;
	kf.statePost.at<float>(3) = bbox.h;
	kf.predict();
	//std::cout << "init=" << kf.statePost << std::endl;
	//std::cout  << std::endl;
}

bbox_t KalmanBoxTracker::predict()
{
	kf.predict();
	//std::cout << "predict=" << kf.statePost << std::endl;
	time_since_update += 1;
	float w = kf.statePost.at<float>(2) * kf.statePost.at<float>(3);
	float h = kf.statePost.at<float>(3);
	bbox_t bbox;
	bbox.x = kf.statePost.at<float>(0) - w / 2;
	bbox.y = kf.statePost.at<float>(1) - h / 2;
	bbox.w = w;
	bbox.h = h;
	/*bbox.track_id = id;
	bbox.obj_id = m_last_bbox.obj_id;
	bbox.prob = m_last_bbox.prob;
	bbox.frames_counter = m_last_bbox.frames_counter;*/
	//std::cout << "predict=" << kf.statePost << std::endl;
	//std::cout << std::endl;
	return bbox;
}

void KalmanBoxTracker::update(bbox_t bbox)
{
	time_since_update = 0;
	cv::Mat correct(4, 1, CV_32F);
	float centre_x = bbox.x + bbox.w / 2.;
	float centre_y = bbox.y + bbox.h / 2.;
	float a = 1.0 * bbox.w / bbox.h;
	correct.at<float>(0) = centre_x;
	correct.at<float>(1) = centre_y;
	correct.at<float>(2) = a;
	correct.at<float>(3) = bbox.h *1.1;
	//std::cout << "update: correct=" << correct << std::endl;
	kf.correct(correct);
	//std::cout << "update: kf.statePost=" << kf.statePost << std::endl;
	//std::cout << std::endl;
}

Sort::Sort()
{
}


Sort::~Sort()
{
}

std::vector<bbox_t> Sort::update(std::vector<bbox_t> detections)
{
	std::vector<bbox_t> return_correct;
	if (detections.size() == 0) return_correct;
	if (trackers.size() == 0)
	{
		auto iter = detections.begin();
		while (iter != detections.end())
		{
			KalmanBoxTracker trk(*iter);
			iter->track_id = trk.id;
			trackers.emplace_back(trk);
			iter++;
		}
		return detections;
	}

	std::vector<bbox_t> trackers_bboxs;
	for (auto iter = trackers.begin(); iter != trackers.end(); iter++)
	{
		bbox_t bbox = iter->predict();
		trackers_bboxs.emplace_back(bbox);
		
	}

	//get predicted locations from existing trackers
	std::vector<std::vector<int>> rults;
	rults = associate_detections_to_trackers(trackers_bboxs, detections);

	std::vector<int> &matched_trks = rults[0];
	std::vector<int> &unmatched_trks = rults[2];

	std::vector<int> &matched_dets = rults[1];
	std::vector<int> &unmatched_dets = rults[3];

	//update matched trackers with assigned detections
	for (int i = 0; i < matched_trks.size(); i++)
	{
		int trk_idx = matched_trks[i];
		int det_idx = matched_dets[i];

		trackers[trk_idx].update(detections[det_idx]);
		detections[det_idx].track_id = trackers[trk_idx].id;
	}

	//create and initialise new trackers for unmatched detections
	for (int i = 0; i < unmatched_dets.size(); i++)
	{
		int det_idx = unmatched_dets[i];

		bbox_t &bbox = detections[det_idx];
		KalmanBoxTracker trk(bbox);
		bbox.track_id = trk.id;
		trackers.emplace_back(trk);
	}

	auto iter = trackers.begin();

	
	while (iter != trackers.end())
	{
		if (iter->time_since_update < 1)
		{
			float w = iter->kf.statePost.at<float>(2) * iter->kf.statePost.at<float>(3);
			float h = iter->kf.statePost.at<float>(3);
			bbox_t bbox;
			bbox.x = iter->kf.statePost.at<float>(0) - w / 2;
			bbox.y = iter->kf.statePost.at<float>(1) - h / 2;
			bbox.w = w;
			bbox.h = h;
			bbox.track_id = iter->id;
			return_correct.emplace_back(bbox);
		}
		
		if (iter->time_since_update > 30)
		{
			iter = trackers.erase(iter);
		}
		else
		{
			iter++;
		}
		
	}
	return return_correct;
}

std::vector<std::vector<int>> Sort::associate_detections_to_trackers(std::vector<bbox_t> predict_bboxs, std::vector<bbox_t> present_bboxs)
{
	int rows = predict_bboxs.size();
	int cols = present_bboxs.size();
	std::vector<std::vector<float>> cost_matrix;

	for (int r = 0; r < rows; r++)
	{
		std::vector<float> vec;
		int pred_x2 = predict_bboxs[r].x + predict_bboxs[r].w;
		int pred_y2 = predict_bboxs[r].y + predict_bboxs[r].h;
		int pred_area = predict_bboxs[r].w * predict_bboxs[r].h;

		for (int c = 0; c < cols; c++)
		{
			int pres_x2 = present_bboxs[c].x + present_bboxs[c].w;
			int pres_y2 = present_bboxs[c].y + present_bboxs[c].h;
			int pres_area = present_bboxs[c].w * present_bboxs[c].h;

			//计算iou
			int inter_x1 = predict_bboxs[r].x > present_bboxs[c].x ? predict_bboxs[r].x : present_bboxs[c].x;
			int inter_y1 = predict_bboxs[r].y > present_bboxs[c].y ? predict_bboxs[r].y : present_bboxs[c].y;

			int inter_x2 = pred_x2 < pres_x2 ? pred_x2 : pres_x2;
			int inter_y2 = pred_y2 < pres_y2 ? pred_y2 : pres_y2;

			int inter_w = inter_x2 - inter_x1;
			int inter_h = inter_y2 - inter_y1;

			inter_w = inter_w > 0 ? inter_w : 0;
			inter_h = inter_h > 0 ? inter_h : 0;

			float inter_area = inter_w * inter_h;

			float overlap = inter_area / (pred_area + pres_area - inter_area);
			vec.emplace_back(1 - overlap);
		}
		cost_matrix.emplace_back(vec);
	}
	return linear_assignment(cost_matrix);
}