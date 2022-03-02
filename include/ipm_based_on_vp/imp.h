#ifndef IPM_BASED_ON_VP_IPM_H_
#define IPM_BASED_ON_VP_IPM_H_

#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

using Vector5d = Eigen::Matrix<double, 5, 1>;

// note: the ipm parts of the ipm_top < vp.y are invalid in ipm result,
// becase the vanishing point is infinity point
struct IPMParam {
  int ipm_img_w;      // after ipm, dst img width 
  int ipm_img_h;      // after ipm, dst img height
  float ipm_top;      // top on origin img
  float ipm_left;     // left on origin img
  float ipm_bottom;   // bottom on origin img
  float ipm_right;    // right on origin img
  float ipm_portion;  // portion = eps / input_img_h, ipm_top = vp.y + eps
  float xlimits[2];   // x limits in ground, m
  float ylimits[2];   // y limits in ground, m

  IPMParam(int w, int h) 
    : ipm_img_w(w), ipm_img_h(h),
      ipm_top(0.f), ipm_left(0.f), 
      ipm_bottom(0.f), ipm_right(0.f), 
      ipm_portion(0.f) {}

  IPMParam(int w, int h, float top, float left, float bottom, float right, float portion) 
  : ipm_img_w(w), ipm_img_h(h), 
    ipm_top(top), ipm_left(left), 
    ipm_bottom(bottom), ipm_right(right), 
    ipm_portion(portion) {}
};

using IPMParamPtr = std::shared_ptr<IPMParam>;

// ref paper: Real time detection of lane markers in urban streets
class IPM {
  public:
    IPM() {}

    ~IPM() {}

    void Initialize(const Eigen::Matrix3d& intrinsics,
                    const Vector5d& dist,
                    const Eigen::Matrix4d& extrinsics,
                    const IPMParamPtr& ipm_param);

    void Initialize(const Eigen::Matrix3d& intrinsics,
                    const Vector5d& dist,
                    const IPMParamPtr& ipm_param,
                    double pitch,
                    double yaw,
                    double height);

    void RunIPM(const cv::Mat& img);

    inline cv::Mat IPMImage() const {
      return img_ipm_;
    }

    inline Eigen::Vector2f VanishingPoint() const {
      return vp_;
    }

  private:
    void ImageToGround(const std::vector<Eigen::Vector3d>& img_pts, 
                       std::vector<Eigen::Vector3d>* ground_pts);

    void GroundToImage(const std::vector<Eigen::Vector3d>& ground_pts, 
                       std::vector<cv::Point3f>* img_pts);

    void GenerateIPMImage(const cv::Mat& img, 
                          const std::vector<cv::Point2f>& img_pts, 
                          bool bilinear = true);

    void CalcVPbyPitchYaw();
    
  private:
    IPMParamPtr ipm_param_; 
    Eigen::Matrix3d intrinsics_;
    Vector5d dist_;
    cv::Mat K_cv_;
    cv::Mat D_cv_;
    Eigen::Matrix3d rot_;
    double pitch_;
    double yaw_;
    double cam_height_;
    Eigen::Vector2f vp_;
    cv::Mat img_ipm_;

}; // class IPM

#endif // IPM_BASED_ON_VP_IPM_H_