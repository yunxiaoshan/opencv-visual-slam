//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/gpu/gpu.hpp>


using namespace cv;
using namespace cv::gpu;

static void download(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

int main( int argc, char** argv )
{

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    list< cv::Point2f > keypoints;

    std::string detectorType = "Feature2D.BRISK";

    float threshold = 0.05;

    // // =============================== Start of CPU Version ===============================
    // vector<cv::KeyPoint> kps;
    //
    // Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
    // detector->set("thres", 100);
    // detector->detect( img_1, kps );
    //
    // int grid_c_num = 10;
    // int grid_r_num = 10;
    // int img_h = img_1.size().height;
    // int img_w = img_1.size().width;
    // bool grid[grid_r_num][grid_c_num];
    //
    // for ( auto kp:kps )
    // {
    //     //map image coordinates to grid
    //     int pt_x = round(kp.pt.x / img_w * grid_c_num);
    //     int pt_y = round(kp.pt.y / img_h * grid_r_num);
    //     if (!grid[pt_y][pt_x]) {
    //         keypoints.push_back( kp.pt );
    //         grid[pt_y][pt_x] = true;
    //     }
    // }
    //
    // vector<cv::Point2f> next_keypoints;
    // vector<cv::Point2f> prev_keypoints;
    // vector<cv::Point2f> back_keypoints;
    //
    // vector<cv::Point2f> img1_keypoints;
    // vector<cv::Point2f> img2_keypoints;
    // for ( auto kp:keypoints )
    //     prev_keypoints.push_back(kp);
    // vector<unsigned char> forward_status;
    // vector<unsigned char> backward_status;
    // vector<float> error;
    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //
    // // forward
    // cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, forward_status, error );
    // // backward
    // cv::calcOpticalFlowPyrLK( img_2, img_1, next_keypoints, back_keypoints, backward_status, error );
    //
    // for (size_t idx = 0; idx < next_keypoints.size(); idx++) {
    //     double pt_dist = norm(back_keypoints[idx] - prev_keypoints[idx]);
    //     if (pt_dist < threshold && forward_status[idx] && backward_status[idx]) {
    //         img1_keypoints.push_back(prev_keypoints[idx]);
    //         img2_keypoints.push_back(next_keypoints[idx]);
    //     }
    // }
    //
    // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    // cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
    //
    // // =============================== End of CPU Version ===============================




    // =============================== Start of GPU Version ===============================
    // Pump up to GPU
    cv::gpu::GpuMat d_frame_0(img_1);
    cv::gpu::GpuMat d_curr_pts;
    cv::gpu::GoodFeaturesToTrackDetector_GPU gpu_detector = GoodFeaturesToTrackDetector_GPU(250, 0.01, 0);
    gpu_detector(d_frame_0, d_curr_pts);

    // Save detected points
    vector<Point2f> kps(d_curr_pts.cols);
    download(d_curr_pts, kps);

    // Initialize grid in jetson board
    int grid_c_num = 10;
    int grid_r_num = 10;
    int img_h = img_1.size().height;
    int img_w = img_1.size().width;
    bool grid[grid_r_num][grid_c_num];

    for ( auto kp:kps )
    {
        // map image coordinates to grid
        int pt_x = round(kp.x / img_w * grid_c_num);
        int pt_y = round(kp.y / img_h * grid_r_num);
        if (!grid[pt_y][pt_x]) {
            keypoints.push_back(kp);
            grid[pt_y][pt_x] = true;
        }
    }

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    vector<cv::Point2f> back_keypoints;

    vector<cv::Point2f> img1_keypoints;
    vector<cv::Point2f> img2_keypoints;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> forward_status;
    vector<unsigned char> backward_status;
    vector<float> error;

    Mat prev_kpts_mat = Mat(1, prev_keypoints.size(), CV_32FC2, (void*)&prev_keypoints[0]);

    cv::gpu::GpuMat d_frame0(img_1);
    cv::gpu::GpuMat d_frame1(img_2);
    cv::gpu::GpuMat d_prevPts(prev_kpts_mat);
    cv::gpu::GpuMat d_nextPts;
    cv::gpu::GpuMat d_backPts;
    cv::gpu::GpuMat d_status;
    cv::gpu::GpuMat d_back_status;
    cv::gpu::PyrLKOpticalFlow d_pyrLK;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    // forward
    d_pyrLK.sparse(d_frame0, d_frame1, d_prevPts, d_nextPts, d_status);
    // backward
	d_pyrLK.sparse(d_frame1, d_frame0, d_nextPts, d_backPts, d_back_status);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // download
    download(d_nextPts, next_keypoints);
    download(d_status, forward_status);
	download(d_backPts, back_keypoints);
	download(d_back_status, backward_status);

    for (size_t idx = 0; idx < next_keypoints.size(); idx++) {
        double pt_dist = norm(back_keypoints[idx] - prev_keypoints[idx]);
        if (pt_dist < threshold && forward_status[idx] && backward_status[idx]) {
            img1_keypoints.push_back(prev_keypoints[idx]);
            img2_keypoints.push_back(next_keypoints[idx]);
        }
    }

    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
    chrono::duration<double> time_used_algo = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    chrono::duration<double> time_used_down = chrono::duration_cast<chrono::duration<double>>( t3-t1 );
    cout<<"LK Flow use time："<<time_used_algo.count()<<" seconds."<<endl;
    cout<<"LK Flow including variable donwloading use time："<<time_used_down.count()<<" seconds."<<endl;

    // =============================== End of GPU Version ===============================




    // visualize all  keypoints
    hconcat(img_1,img_2,img_1);
    // for ( int i=0; i< prev_keypoints.size() ;i++)
    for ( unsigned int i=0; i< img1_keypoints.size() ;i++)
    {
        // cout<<(int)forward_status[i]<<endl;
        if(forward_status[i] == 1)
        {
            Point pt;
            // pt.x =  next_keypoints[i].x + img_2.size[1];
            // pt.y =  next_keypoints[i].y;

            // line(img_1, prev_keypoints[i], pt, cv::Scalar(0,255,255));
            pt.x =  img2_keypoints[i].x + img_2.size[1];
            pt.y =  img2_keypoints[i].y;

            line(img_1, img1_keypoints[i], pt, cv::Scalar(0,255,255));
        }
    }

    cv::imshow("klt tracker", img_1);
    cv::waitKey(0);

    return 0;
}
