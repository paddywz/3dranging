#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class BinoCameraCalibrator final {
public:
    BinoCameraCalibrator() = delete;
    BinoCameraCalibrator(const BinoCameraCalibrator &) = delete;
    BinoCameraCalibrator(BinoCameraCalibrator &&) = delete;
    BinoCameraCalibrator &operator=(const BinoCameraCalibrator &) = delete;
    BinoCameraCalibrator &operator=(BinoCameraCalibrator &&) = delete;

public:
    BinoCameraCalibrator(const std::vector<std::string> &leftImageFiles,
                         const std::vector<std::string> &rightImageFiles,
                         const std::vector<cv::Point3f> &objp);
    cv::Matx33f GetLeftCameraMatrix() const;
    cv::Matx33f GetRightCameraMatrix() const;
    cv::Vec<float, 5> GetLeftDistCoeffs() const;
    cv::Vec<float, 5> GetRightDistCoeffs() const;
    std::vector<cv::Mat> GetLeftRVecs() const;
    std::vector<cv::Mat> GetRightRVecs() const;
    std::vector<cv::Mat> GetLeftTvecs() const;
    std::vector<cv::Mat> GetRightTvecs() const;

    cv::Mat GetBinoRMatrix() const;
    cv::Mat GetBinoTMatrix() const;
    cv::Mat GetBinoEMatrix() const;
    cv::Mat GetBinoFMatrix() const;

    cv::Mat GetBinoQMatrix() const;
    cv::Mat GetBinoLeftRMatrix() const;
    cv::Mat GetBinoRightRMatrix() const;
    cv::Mat GetBinoLeftPMatrix() const;
    cv::Mat GetBinoRightPMatrix() const;

    void Calibrate(const std::vector<std::string> &fileNames,
                   std::vector<std::vector<cv::Point3f>> &objectPoints,
                   std::vector<std::vector<cv::Point2f>> &imagePoints,
                   cv::Matx33f &cameraMatrix, cv::Vec<float, 5> &disCoeffs,
                   std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs);
    void BinoCalibrate();
    void StereoCalibrate();
    void SteroRectify();

    bool SaveAllInfo(const std::string &path);

private:
    const std::vector<std::string> m_leftImageFiles;
    const std::vector<std::string> m_rightImageFiles;
    const std::vector<cv::Point3f> m_objp;

    cv::Matx33f m_leftCameraMatrix;
    cv::Matx33f m_rightCameraMatrix;
    cv::Vec<float, 5> m_leftDistCoeffs;
    cv::Vec<float, 5> m_rightDistCoeffs;

    std::vector<cv::Mat> m_leftRVecs;
    std::vector<cv::Mat> m_rightRVecs;
    std::vector<cv::Mat> m_leftTVecs;
    std::vector<cv::Mat> m_rightTVecs;

    cv::Mat m_binoR;
    cv::Mat m_binoT;
    cv::Mat m_binoE;
    cv::Mat m_binoF;

    cv::Mat m_binoQ;
    cv::Mat m_binoLeftR;
    cv::Mat m_binoRightR;
    cv::Mat m_binoLeftP;
    cv::Mat m_binoRightP;

    std::vector<std::vector<cv::Point3f>> m_objectPoints;
    std::vector<std::vector<cv::Point2f>> m_leftImagePoints;
    std::vector<std::vector<cv::Point2f>> m_rightImagePoints;

    int m_checkBoardRow;
    int m_checkBoardCol;
};
