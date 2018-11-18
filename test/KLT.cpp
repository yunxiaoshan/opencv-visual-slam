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

using namespace cv;
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
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
	detector->set("thres", 100);


    detector->detect( img_1, kps );

    // TODO Initialize grid
    // std::cout << img_1.rows << " " << img_1.cols << std::endl;
    int grid_c_size = 16;
    int grid_r_size = 12;
    int img_h = img_1.rows;
    int img_w = img_1.cols;
    bool grid[grid_r_size][grid_c_size];

    for ( auto kp:kps )
    {
        // map image coordinates to grid
        int pt_x = round(kp.pt.x / img_w * grid_c_size);
        int pt_y = round(kp.pt.y / img_h * grid_r_size);
        if (!grid[pt_y][pt_x]) {
            std::cout << kp.pt << std::endl;
            keypoints.push_back( kp.pt );
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

    Mat prev_kpts_mat = Mat(1, prev_keypoints.size(), CV_32FC2);
	for (size_t i = 0; i < prev_keypoints.size(); i++) {
		prev_kpts_mat.at<Vec2f>(0, i)[0] = prev_keypoints[i].x;
		prev_kpts_mat.at<Vec2f>(0, i)[1] = prev_keypoints[i].y;
	}

    GpuMat d_frame0(img_1);
    GpuMat d_frame1(img_2);
    GpuMat d_prevPts(prev_kpts_mat);
    GpuMat d_nextPts;
    GpuMat d_backPts;
    GpuMat d_status;
    GpuMat d_back_status;
    PyrLKOpticalFlow d_pyrLK;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    // // ====== CPU Version =======
    // // forward
    // cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, forward_status, error );
    // // backward
    // cv::calcOpticalFlowPyrLK( img_2, img_1, next_keypoints, back_keypoints, backward_status, error );

    // ====== GPU Version =======
    // forward
    d_pyrLK.sparse(d_frame0, d_frame1, d_prevPts, d_nextPts, d_status);
    downloadpts(d_nextPts, next_keypoints);
	downloadmask(d_status, forward_status);
    // backward
	d_pyrLK.sparse(d_frame1, d_frame0, d_nextPts, d_backPts, d_back_status);
	downloadpts(d_backPts, back_keypoints);
	downloadmask(d_back_status, backward_status);

    for (size_t idx = 0; idx < next_keypoints.size(); idx++) {
        double pt_dist = norm(back_keypoints[idx] - prev_keypoints[idx]);
        if (pt_dist < 0.01 && forward_status[idx] && backward_status[idx]) {
            img1_keypoints.push_back(prev_keypoints[idx]);
            img2_keypoints.push_back(next_keypoints[idx]);
        }
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    // visualize all  keypoints
    hconcat(img_1,img_2,img_1);
    // for ( int i=0; i< prev_keypoints.size() ;i++)
    for ( unsigned int i=0; i< img1_keypoints.size() ;i++)
    {
        cout<<(int)forward_status[i]<<endl;
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
