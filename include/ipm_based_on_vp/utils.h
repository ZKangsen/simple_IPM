#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>

class Utils {
  public:
    template<typename T>
    static void UndistortPoints(const cv::Mat& K, 
                                const cv::Mat& D, 
                                const std::vector<cv::Point_<T>>& pts,
                                std::vector<cv::Point2f>* undist_pts) {
      if (fabs(D.at<double>(0)) < 1e-4) {
        for (const auto& pt : pts) {
          undist_pts->emplace_back(pt.x, pt.y);
        }
        return;
      }

      size_t num = pts.size();
      cv::Mat pts_mat(num, 2, CV_32F);
      for(int i = 0; i < num; ++i) {
        pts_mat.ptr<float>(i)[0] = static_cast<float>(pts[i].x);
        pts_mat.ptr<float>(i)[1] = static_cast<float>(pts[i].y);
      }

      pts_mat = pts_mat.reshape(2);
      cv::undistortPoints(pts_mat, pts_mat, K, D, cv::Mat(), K);
      pts_mat = pts_mat.reshape(1);

      for(int i = 0; i < num; ++i) {
        undist_pts->emplace_back(pts_mat.ptr<float>(i)[0],
                                 pts_mat.ptr<float>(i)[1]);
      }

      return;
    }

    static void ProjectPoints(const cv::Mat& K, 
                              const cv::Mat& D, 
                              const std::vector<cv::Point3f>& pts,
                              std::vector<cv::Point2f>* dist_pts) {
      cv::Mat R = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
                                             0.0, 1.0, 0.0,
                                             0.0, 0.0, 1.0);
      cv::Mat t = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);

      cv::projectPoints(pts, R, t, K, D, *dist_pts);
      
      return;
    }

    static bool CheckInBorder(const cv::Point2f& pt, int width, int height) {
      static const int margin = 1;
      int x = cvRound(pt.x);
      int y = cvRound(pt.y);
      if (x <= margin || x >= width - margin || 
          y <= margin || y >= height - margin) {
        return false;
      }
      return true;
    }

}; // class Utils

#endif // UTILS_H_