#pragma once
#include <opencv.hpp>
#include <vector>
#include "yolo_v2_class.hpp"



class Sort;
class KalmanBoxTracker
{
	friend class Sort;
public:
	KalmanBoxTracker(bbox_t bbox);
	~KalmanBoxTracker();
	static int count;

public:
	void initialize_kalman(bbox_t bbox);
	bbox_t predict();
	void update(bbox_t bbox);

private:
	//bbox_t m_last_bbox;
	cv::KalmanFilter kf;
	int time_since_update = 0;
	int id = -1;
	cv::Mat meas;
};




class Sort
{
public:
	Sort();
	~Sort();
public:
	std::vector<bbox_t> update(std::vector<bbox_t> detections);
	std::vector<std::vector<int>> associate_detections_to_trackers(std::vector<bbox_t> predict_bboxs, std::vector<bbox_t> present_bboxs);
private:
	std::vector<KalmanBoxTracker> trackers;
};

