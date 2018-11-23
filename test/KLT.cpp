////
//// Created by sicong on 08/11/18.
////
//
//#include <iostream>
//#include <fstream>
//#include <list>
//#include <vector>
//#include <chrono>
//using namespace std;
//
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/video/tracking.hpp>
//
//using namespace cv;
//int main( int argc, char** argv )
//{
//
//    if ( argc != 3 )
//    {
//        cout<<"usage: feature_extraction img1 img2"<<endl;
//        return 1;
//    }
//    //-- Read two images
//    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
//    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
//
//    list< cv::Point2f > keypoints;
//    vector<cv::KeyPoint> kps;
//
//    std::string detectorType = "Feature2D.BRISK";
//    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", 100);
//
//
//    detector->detect( img_1, kps );
//    for ( auto kp:kps )
//        keypoints.push_back( kp.pt );
//
//    vector<cv::Point2f> next_keypoints;
//    vector<cv::Point2f> prev_keypoints;
//    for ( auto kp:keypoints )
//        prev_keypoints.push_back(kp);
//    vector<unsigned char> status;
//    vector<float> error;
//    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
//    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
//    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
//
//    // visualize all  keypoints
//    hconcat(img_1,img_2,img_1);
//    for ( int i=0; i< prev_keypoints.size() ;i++)
//    {
//        cout<<(int)status[i]<<endl;
//        if(status[i] == 1)
//        {
//            Point pt;
//            pt.x =  next_keypoints[i].x + img_2.size[1];
//            pt.y =  next_keypoints[i].y;
//
//            line(img_1, prev_keypoints[i], pt, cv::Scalar(0,255,255));
//        }
//    }
//
//    cv::imshow("klt tracker", img_1);
//    cv::waitKey(0);
//
//    return 0;
//}

//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <math.h>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

cv::Matx33d Findfundamental(vector<cv::Point2f> prev_subset, vector<cv::Point2f> next_subset,int img_w, int img_h)
{
    // TODO
    // normalize the points
    cv::Mat norm = (Mat_<double>(3,3)<<2.0/img_w, 0, -1, 0, 2.0/img_h, -1, 0, 0, 1); 

    // get the points
    int num_vals = prev_subset.size() + 1;
    // 8 * 9, the last row should be 0
    cv::Mat W = cv::Mat::zeros(prev_subset.size(), num_vals, CV_64F);
    for (size_t i = 0; i < prev_subset.size(); i++)
    {
        cv::Mat prev = (Mat_<double>(3,1)<<prev_subset[i].x, prev_subset[i].y, 1); 
        cv::Mat next = (Mat_<double>(3,1)<<next_subset[i].x, next_subset[i].y, 1); 

        cv::Mat prev_norm = norm * prev;
        cv::Mat next_norm = norm * next;

        double u1 = prev_norm.at<double>(0,0);
        double v1 = prev_norm.at<double>(1,0);
        double u2 = next_norm.at<double>(0,0);
        double v2 = next_norm.at<double>(1,0);
        double curr_point[] = {u1 * u2, u1 * v2, u1, v1 * u2, v1 * v2, v1, u2, v2, 1.0};
        cv::Mat curr_row = cv::Mat(1, num_vals, CV_64F, curr_point);
        curr_row.copyTo(W.row(i));
    }

    // first SVD
    cv::SVD svd1(W); // get svd.vt, svd.w, svd.u;

    // second SVD
    cv::Mat e_hat = cv::Mat(3, 3, CV_64F);

    for (int i = 0; i < 9; i++)
    {
        e_hat.at<double>(i/3, i%3) = svd1.vt.at<double>((num_vals - 2), i);
    }

    cv::SVD svd2(e_hat);
    cv::Mat w;
    svd2.w.copyTo(w);
    
    cv::Mat w_hat = cv::Mat::zeros(3,3, CV_64F);
    w_hat.at<double>(0, 0) = w.at<double>(0, 0);
    w_hat.at<double>(1, 1) = w.at<double>(1, 0);

    cv::Mat F_hat = svd2.u * w_hat * svd2.vt;

    // TODO
    // denormalize points
    cv::Mat F_norm = norm.t()*F_hat*norm;
    cv::Matx33d F((double *)F_norm.clone().ptr());

    return F;
}
bool checkinlier(cv::Point2f prev_keypoint, cv::Point2f next_keypoint, cv::Matx33d Fcandidate, double d)
{
    double u1 = prev_keypoint.x;
    double v1 = prev_keypoint.y;
    double u2 = next_keypoint.x;
    double v2 = next_keypoint.y;

    // epipolar line 1 to 2
    cv::Matx33d Fcandidate_t = Fcandidate.t();
    // cout << "F transpose " << Fcandidate_t << endl;
    double a2 = Fcandidate_t(0, 0) * u1 + Fcandidate_t(0, 1) * v1 + Fcandidate_t(0, 2);
    double b2 = Fcandidate_t(1, 0) * u1 + Fcandidate_t(1, 1) * v1 + Fcandidate_t(1, 2);
    double c2 = Fcandidate_t(2, 0) * u1 + Fcandidate_t(2, 1) * v1 + Fcandidate_t(2, 2);

    double dist = (double)abs(a2 * u2 + b2 * v2 + c2) / sqrt(a2 * a2 + b2 * b2);
    cout << "dist " << dist << endl;
    return dist <= d;
}

int main(int argc, char **argv)
{

    srand(time(NULL));

    if (argc != 3)
    {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    // Get input img size
    int img_w = img_1.size().width;
    int img_h = img_1.size().height;

    list<cv::Point2f> keypoints;
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector> detector = Algorithm::create<FeatureDetector>(detectorType);
    detector->set("thres", 100);

    detector->detect(img_1, kps);
    for (auto kp : kps)
        keypoints.push_back(kp.pt);

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for (auto kp : keypoints)
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img_1, img_2, prev_keypoints, next_keypoints, status, error);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "LK Flow use time：" << time_used.count() << " seconds." << endl;

    vector<cv::Point2f> kps_prev, kps_next;
    kps_prev.clear();
    kps_next.clear();
    for (size_t i = 0; i < prev_keypoints.size(); i++)
    {
        if (status[i] == 1)
        {
            kps_prev.push_back(prev_keypoints[i]);
            kps_next.push_back(next_keypoints[i]);
        }
    }

    // p Probability that at least one valid set of inliers is chosen
    // d Tolerated distance from the model for inliers
    // e Assumed outlier percent in data set.
    double p = 0.99;
    double d = 1.5f;
    double e = 0.2;

    int niter = static_cast<int>(std::ceil(std::log(1.0 - p) / std::log(1.0 - std::pow(1.0 - e, 8))));
    Mat Fundamental;
    cv::Matx33d F, Fcandidate;
    int bestinliers = -1;
    vector<cv::Point2f> prev_subset, next_subset;
    int matches = kps_prev.size();
    prev_subset.clear();
    next_subset.clear();

    for (int i = 0; i < niter; i++)
    {
        // step1: randomly sample 8 matches for 8pt algorithm
        unordered_set<int> rand_util;
        while (rand_util.size() < 8)
        {
            int randi = rand() % matches;
            rand_util.insert(randi);
        }
        vector<int> random_indices(rand_util.begin(), rand_util.end());
        for (size_t j = 0; j < rand_util.size(); j++)
        {
            prev_subset.push_back(kps_prev[random_indices[j]]);
            next_subset.push_back(kps_next[random_indices[j]]);
        }
        // step2: perform 8pt algorithm, get candidate F

        Fcandidate = Findfundamental(prev_subset, next_subset, img_w, img_h);
        // step3: Evaluate inliers, decide if we need to update the best solution
        int inliers = 0;
        for (size_t j = 0; j < kps_prev.size(); j++)
        {
            if (checkinlier(kps_prev[j], kps_next[j], Fcandidate, d))
                inliers++;
        }
        if (inliers > bestinliers)
        {
            F = Fcandidate;
            bestinliers = inliers;
        }
        prev_subset.clear();
        next_subset.clear();
    }

    // step4: After we finish all the iterations, use the inliers of the best model to compute Fundamental matrix again.

    for (size_t j = 0; j < kps_prev.size(); j++)
    {
        if (checkinlier(kps_prev[j], kps_next[j], F, d))
        {
            prev_subset.push_back(kps_prev[j]);
            next_subset.push_back(kps_next[j]);
        }
    }
	cout << kps_prev.size()<< endl;
	cout << next_subset.size()<< endl;

    F = Findfundamental(prev_subset, next_subset, img_w, img_h);

    cout << "Fundamental matrix is \n"
         << F << endl;
    return 0;
}
