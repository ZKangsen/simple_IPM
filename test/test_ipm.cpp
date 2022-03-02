#include <string>
#include <glog/logging.h>
#include "ipm_based_on_vp/imp.h"

int main(int argc, char** argv) {
  // init log
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  Eigen::Matrix3d intrinsics;
  Vector5d dist;
  Eigen::Matrix4d extrinsics(Eigen::Matrix4d::Identity());

  intrinsics << 1968.1435602859058, 0, 935.73824552105441,
               0, 1969.0276646548698, 546.28824947904047,
               0,                  0,                  1;

  dist << -0.57787296417449807,
          0.3452587138288295,
          0.0012622364846943547,
          0.00034294071465058873,
          -0.10211367100931869;
  
  extrinsics.block<3, 3>(0, 0) = 
    Eigen::Quaterniond(-0.49399897413475441,
                        0.50082395694283666,
                       -0.49746997674813787,
                        0.5076061464759235).toRotationMatrix();
  extrinsics.block<3, 1>(0, 3) = 
    Eigen::Vector3d(1.5192374447437267,
                    0.43397530377787419,
                    1.5837838717498334);
  
  std::string file = "../imgs/test.png";
  cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
  int ipm_width = 640;
  int ipm_height = 480;
  float ipm_top = 0.f;
  float ipm_left = 0.f;
  float ipm_bottom = img.rows - 1; 
  float ipm_right = img.cols - 1; 
  float ipm_portion = 0.1;
  IPMParamPtr ipm_param(new IPMParam(ipm_width, ipm_height, 
                                     ipm_top, ipm_left, 
                                     ipm_bottom, ipm_right, 
                                     ipm_portion));
  
  IPM ipm;
  ipm.Initialize(intrinsics, dist, extrinsics, ipm_param);
  ipm.RunIPM(img);
  Eigen::Vector2f vp = ipm.VanishingPoint();
  cv::Mat ipm_img = ipm.IPMImage();

  LOG(INFO) << "vanishing point: " << vp.transpose();
  LOG(INFO) << "ipm xlimits: " << ipm_param->xlimits[0] << " " << ipm_param->xlimits[1];
  LOG(INFO) << "ipm ylimits: " << ipm_param->ylimits[0] << " " << ipm_param->ylimits[1];
  // cv::imshow("input img", img);
  cv::imshow("ipm img", ipm_img);
  cv::waitKey(0);

  return 0;
}
