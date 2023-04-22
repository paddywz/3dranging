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
    cv::Mat GetLeftCameraMatrix() const;
    cv::Mat GetRightCameraMatrix() const;
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
    cv::Mat GetBinoLeftPMatrix() const;
    cv::Mat GetBinoRightPMatrix() const;

    void Calibrate(std::vector<std::string> &fileNames);
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
    cv::Mat m_binoLeftP;
    cv::Mat m_binoRightP;

    int m_checkBoardRow;
    int m_checkBoardCol;
};
