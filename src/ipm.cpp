#include "ipm_based_on_vp/imp.h"
#include <cmath>
#include <algorithm>
#include <glog/logging.h>
#include "ipm_based_on_vp/utils.h"

void IPM::Initialize(const Eigen::Matrix3d& intrinsics, 
                     const Vector5d& dist,
                     const Eigen::Matrix4d& extrinsics,
                     const IPMParamPtr& ipm_param) {
  intrinsics_ = intrinsics;
  dist_ = dist;
  K_cv_ =
    (cv::Mat_<double>(3, 3) << intrinsics(0, 0), intrinsics(0, 1), intrinsics(0, 2),
                               intrinsics(1, 0), intrinsics(1, 1), intrinsics(1, 2),
                               intrinsics(2, 0), intrinsics(2, 1), intrinsics(2, 2));
  D_cv_ =
    (cv::Mat_<double>(5, 1) << dist(0), dist(1), dist(2), dist(3), dist(4));
  ipm_param_ = ipm_param;
  rot_ = extrinsics.block<3, 3>(0, 0);
  Eigen::Vector3d ypr = rot_.eulerAngles(2, 1, 0); // Z-Y-X, External rotation
  pitch_ = ypr[1];
  yaw_ = ypr[0];
  // rot_ = 
  //   (Eigen::AngleAxisd(yaw_,   Eigen::Vector3d::UnitZ()) *
  //    Eigen::AngleAxisd(pitch_, Eigen::Vector3d::UnitY()) *
  //    Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX())).toRotationMatrix();
  cam_height_ = extrinsics(2, 3);
  LOG(INFO) << "IPM initialized successfully...";
  return;
}

void IPM::Initialize(const Eigen::Matrix3d& intrinsics, 
                     const Vector5d& dist,
                     const IPMParamPtr& ipm_param,
                     double pitch,
                     double yaw,
                     double height) {
  intrinsics_ = intrinsics;
  dist_ = dist;
  K_cv_ =
    (cv::Mat_<double>(3, 3) << intrinsics(0, 0), intrinsics(0, 1), intrinsics(0, 2),
                               intrinsics(1, 0), intrinsics(1, 1), intrinsics(1, 2),
                               intrinsics(2, 0), intrinsics(2, 1), intrinsics(2, 2));
  D_cv_ =
    (cv::Mat_<double>(5, 1) << dist(0), dist(1), dist(2), dist(3), dist(4));
  ipm_param_ = ipm_param;
  rot_ = 
    (Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ()) *
     Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())).toRotationMatrix();
  pitch_ = pitch;
  yaw_ = yaw;
  cam_height_ = height;
  LOG(INFO) << "IPM initialized successfully...";
  return;
}

void IPM::RunIPM(const cv::Mat& img) {
  CalcVPbyPitchYaw();

  int img_w = img.cols;
  int img_h = img.rows;
  float eps = img_h * ipm_param_->ipm_portion;
  ipm_param_->ipm_top = std::max(ipm_param_->ipm_top, vp_.y() + eps);
  ipm_param_->ipm_left = std::max(ipm_param_->ipm_left, 0.f);
  ipm_param_->ipm_bottom = std::min(ipm_param_->ipm_bottom, img_h - 1.f);
  ipm_param_->ipm_right = std::min(ipm_param_->ipm_right, img_w - 1.f);
  
  std::vector<Eigen::Vector3d> img_pts;
  img_pts.emplace_back(ipm_param_->ipm_left, ipm_param_->ipm_top, 1.0);
  img_pts.emplace_back(vp_.x(), ipm_param_->ipm_top, 1.0);
  img_pts.emplace_back(ipm_param_->ipm_right, ipm_param_->ipm_top, 1.0);
  img_pts.emplace_back(vp_.x(), ipm_param_->ipm_bottom, 1.0);

  std::vector<Eigen::Vector3d> ground_pts;
  ImageToGround(img_pts, &ground_pts);
  double min_x, max_x, min_y, max_y;
  std::sort(ground_pts.begin(), ground_pts.end(), 
    [](const Eigen::Vector3d& p1, const Eigen::Vector3d& p2){
      return p1.x() > p2.x();
    });
  min_x = ground_pts.back().x();
  max_x = ground_pts.front().x();
  std::sort(ground_pts.begin(), ground_pts.end(), 
    [](const Eigen::Vector3d& p1, const Eigen::Vector3d& p2){
      return p1.y() > p2.y();
    });
  min_y = ground_pts.back().y();
  max_y = ground_pts.front().y();

  double x_range = max_x - min_x;
  double y_range = max_y - min_y;
  double x_step = x_range / ipm_param_->ipm_img_h;
  double y_step = y_range / ipm_param_->ipm_img_w;

  std::vector<Eigen::Vector3d> ground_grid;
  ground_grid.reserve(ipm_param_->ipm_img_h * ipm_param_->ipm_img_w);
  for (int i = 0; i < ipm_param_->ipm_img_h; ++i) {
    for (int j = 0; j < ipm_param_->ipm_img_w; ++j) {
      ground_grid.emplace_back(max_x - (i + 0.5) * x_step, 
                               max_y - (j + 0.5) * y_step, 
                               -cam_height_);
    }
  }
  std::vector<cv::Point3f> img_grid;
  img_grid.reserve(ground_grid.size());
  GroundToImage(ground_grid, &img_grid);

  // distort the normalized pts to origin img pts
  std::vector<cv::Point2f> img_grid_dist;
  img_grid_dist.reserve(ground_grid.size());
  Utils::ProjectPoints(K_cv_, D_cv_, img_grid, &img_grid_dist);
  GenerateIPMImage(img, img_grid_dist);

  ipm_param_->xlimits[0] = min_x;
  ipm_param_->xlimits[1] = max_x;
  ipm_param_->ylimits[0] = min_y;
  ipm_param_->ylimits[1] = max_y;

  return;
}

void IPM::ImageToGround(const std::vector<Eigen::Vector3d>& img_pts, 
                        std::vector<Eigen::Vector3d>* ground_pts) {
  Eigen::Matrix3d intrinsics_inv = intrinsics_.inverse();
  for (const auto& pt : img_pts) {
    Eigen::Vector3d pt_cache = rot_ * intrinsics_inv * pt;
    double cam_z = -cam_height_ / pt_cache.z();
    pt_cache *= cam_z;
    ground_pts->emplace_back(pt_cache);
  }
  return;
}

void IPM::GroundToImage(const std::vector<Eigen::Vector3d>& ground_pts, 
                        std::vector<cv::Point3f>* img_pts) {
  for (const auto& pt : ground_pts) {
    Eigen::Vector3d pt_cache = rot_.transpose() * pt;
    pt_cache /= pt_cache.z();
    Eigen::Vector3f pt_f = pt_cache.cast<float>();
    // pt_f is normalized pts
    img_pts->emplace_back(pt_f.x(), pt_f.y(), pt_f.z());
  }
  return;
}

// ref blog: https://blog.csdn.net/yeyang911/article/details/51912322
// world frame: FLU(front-left-up)
// camera frame: RBF(right-bottom-front)
// assume only rotation exist between world frame and camera frame.
void IPM::CalcVPbyPitchYaw() {
  Eigen::Vector3d vp_on_ground(1, 0, 0);  // direction vector in world frame
  Eigen::Vector3d vp_on_image = intrinsics_ * rot_.transpose() * vp_on_ground;
  vp_on_image /= vp_on_image.z();
  vp_ = vp_on_image.head(2).cast<float>();
  return;
}

void IPM::GenerateIPMImage(const cv::Mat& img, 
                           const std::vector<cv::Point2f>& img_pts, 
                           bool bilinear) {
  img_ipm_ = cv::Mat::zeros(ipm_param_->ipm_img_h, ipm_param_->ipm_img_w, CV_8UC1);
  for (int i = 0; i < ipm_param_->ipm_img_h; ++i) {
    uint8_t* data_ptr = img_ipm_.ptr<uint8_t>(i);
    for (int j = 0; j < ipm_param_->ipm_img_w; ++j) {
      cv::Point2f pt = img_pts[i * ipm_param_->ipm_img_w + j];
      if (pt.x < ipm_param_->ipm_left || pt.x > ipm_param_->ipm_right || 
          pt.y < ipm_param_->ipm_top || pt.y > ipm_param_->ipm_bottom) {
        continue;
      }
      // interpolate
      float value = 0;
      if (bilinear) {               // bilinear interpolation
        float du = pt.x - int(pt.x);
        float dv = pt.y - int(pt.y);
        const uint8_t* data = &img.data[int(pt.y) * img.step + int(pt.x)];
        value = (1 - du) * (1 - dv) * data[0] + 
                du * (1 - dv) * data[1] + 
                (1 - du) * dv * data[img.step] + 
                du * dv * data[img.step + 1];
      } else {                      // nearest interpolation
        int u = cvRound(pt.x);
        int v = cvRound(pt.y);
        value = img.ptr<uint8_t>(v)[u];
      }
      data_ptr[j] = static_cast<uint8_t>(value);
    }
  }
}
