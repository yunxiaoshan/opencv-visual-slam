#include "frontend/VisualFrontend.h"
#include <chrono>
using namespace std;
using namespace cv;

void VisualFrontend::downloadpts(const GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

void VisualFrontend::downloadmask(const GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

VisualFrontend::VisualFrontend()
{
	//Initialise detector
//	std::string detectorType = "Feature2D.BRISK";
//
//	detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", thresholdExtraction);
	//Initialize ID
    gpu_detector = GoodFeaturesToTrackDetector_GPU(250, 0.01, 0);

    d_pyrLK.winSize.width = 21;
    d_pyrLK.winSize.height = 21;
    d_pyrLK.maxLevel = 3;
    d_pyrLK.iters = 30;

	newId = 0;

}

void VisualFrontend::trackAndExtract(cv::Mat& im_gray, Features2D& trackedPoints, Features2D& newPoints)
{
	if (oldPoints.size() > 0)
	{
        //Track prevoius points with optical flow
		auto festart = chrono::steady_clock::now();
		track1(im_gray, trackedPoints);
		auto feend = chrono::steady_clock::now();
		cout << "klt running time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;


		//Save tracked points
		oldPoints = trackedPoints;
	}

	//Extract new points
	auto festart = chrono::steady_clock::now();
	extract1(im_gray, newPoints);
	auto feend = chrono::steady_clock::now();
	cout << "new feature time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;

	//save old image
	im_prev = im_gray;

}

void VisualFrontend::extract1(Mat& im_gray, Features2D& newPoints)
{
    // TODO
	// reset grid with old feature points
	vector<Point2f> oldFeaturePoints = oldPoints.getPoints();
	grid.setImageSize1(im_gray.cols, im_gray.rows);
	for (size_t i = 0; i < oldFeaturePoints.size(); i++) {
		grid.addPoint1(oldFeaturePoints[i]);
	}

	// Pump up to GPU
	GpuMat d_frame_curr(im_gray);
	GpuMat d_curr_pts;
	gpu_detector(d_frame_curr, d_curr_pts);

	// Save detected points
	vector<Point2f> newFeaturePoints(d_curr_pts.cols);
	downloadpts(d_curr_pts, newFeaturePoints);

	// if new points, add
	for (size_t i = 0; i < newFeaturePoints.size(); i++) {
		Point2f curr_pt = newFeaturePoints[i];
		if (grid.isNewFeature1(curr_pt)) {
			oldPoints.addPoint(curr_pt, newId);
			newPoints.addPoint(curr_pt, newId);
			newId++;
		}
	}

	// reset grid
	grid.resetGrid1();
}

void VisualFrontend::track1(Mat& im_gray, Features2D& trackedPoints)
{
    // TODO

	// old features to cv mat to gpu mat
	vector<Point2f> prevPts = oldPoints.getPoints();
	Mat oldCVPts = Mat(1, prevPts.size(), CV_32FC2);
	for (size_t i = 0; i < prevPts.size(); i++) {
		oldCVPts.at<Vec2f>(0, i)[0] = prevPts[i].x;
		oldCVPts.at<Vec2f>(0, i)[1] = prevPts[i].y;
	}
	//~ Mat<Point2f> oldCVPts = Mat(prevPts);

	// init gpu variables
	GpuMat d_frame0(im_prev);
	GpuMat d_frame1(im_gray);
	GpuMat d_prevPts(oldCVPts);
	GpuMat d_nextPts;
	GpuMat d_backPts;
	GpuMat d_status;
	GpuMat d_back_status;

	// forward
	d_pyrLK.sparse(d_frame0, d_frame1, d_prevPts, d_nextPts, d_status);

	vector<Point2f> nextPts(d_nextPts.cols);
	vector<unsigned char> forwardStatus(d_status.cols);
	downloadpts(d_nextPts, nextPts);
	downloadmask(d_status, forwardStatus);

	// backward
	d_pyrLK.sparse(d_frame1, d_frame0, d_nextPts, d_backPts, d_back_status);

	vector<Point2f> backPts(d_backPts.cols);
	vector<unsigned char> backwardStatus(d_status.cols);
	downloadpts(d_backPts, backPts);
	downloadmask(d_back_status, backwardStatus);

	// Compare distance between prev and back
	vector<unsigned char> status(forwardStatus.size());
	for (size_t idx = 0; idx < backPts.size(); idx++) {
        double pt_dist = norm(backPts[idx] - prevPts[idx]);
        status[idx] = (pt_dist < thresholdFBError) && forwardStatus[idx] && backwardStatus[idx];
    }

	trackedPoints = Features2D(oldPoints, nextPts, status);
}
