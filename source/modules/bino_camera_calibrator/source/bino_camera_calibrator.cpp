#include <3dranging/bino_camera_calibrator.h>
#include <cstddef>
#include <vector>

BinoCameraCalibrator::BinoCameraCalibrator(
    const std::vector<std::string> &leftImageFiles,
    const std::vector<std::string> &rightImageFiles,
    const std::vector<cv::Point3f> &objp)
    : m_leftImageFiles(leftImageFiles),
      m_rightImageFiles(rightImageFiles),
      m_objp(objp),
      m_leftCameraMatrix(cv::Matx33f::eye()),
      m_rightCameraMatrix(cv::Matx33f::eye()),
      m_leftDistCoeffs(0, 0, 0, 0, 0),
      m_rightDistCoeffs(0, 0, 0, 0, 0),
      m_leftImagePoints(leftImageFiles.size()),
      m_rightImagePoints(rightImageFiles.size()) {}

cv::Matx33f BinoCameraCalibrator::GetLeftCameraMatrix() const {
    return m_leftCameraMatrix;
}

cv::Matx33f BinoCameraCalibrator::GetRightCameraMatrix() const {
    return m_rightCameraMatrix;
}

cv::Vec<float, 5> BinoCameraCalibrator::GetLeftDistCoeffs() const {
    return m_leftDistCoeffs;
}

cv::Vec<float, 5> BinoCameraCalibrator::GetRightDistCoeffs() const {
    return m_rightDistCoeffs;
}

std::vector<cv::Mat> BinoCameraCalibrator::GetLeftRVecs() const {
    return m_leftRVecs;
}

std::vector<cv::Mat> BinoCameraCalibrator::GetRightRVecs() const {
    return m_rightRVecs;
}

std::vector<cv::Mat> BinoCameraCalibrator::GetLeftTvecs() const {
    return m_leftTVecs;
}

std::vector<cv::Mat> BinoCameraCalibrator::GetRightTvecs() const {
    return m_rightTVecs;
}

cv::Mat BinoCameraCalibrator::GetBinoRMatrix() const { return m_binoR; }

cv::Mat BinoCameraCalibrator::GetBinoTMatrix() const { return m_binoT; }

cv::Mat BinoCameraCalibrator::GetBinoEMatrix() const { return m_binoE; }

cv::Mat BinoCameraCalibrator::GetBinoFMatrix() const { return m_binoF; }

cv::Mat BinoCameraCalibrator::GetBinoQMatrix() const { return m_binoQ; }

cv::Mat BinoCameraCalibrator::GetBinoLeftRMatrix() const { return m_binoLeftR; }

cv::Mat BinoCameraCalibrator::GetBinoRightRMatrix() const {
    return m_binoRightR;
}

cv::Mat BinoCameraCalibrator::GetBinoLeftPMatrix() const { return m_binoLeftP; }

cv::Mat BinoCameraCalibrator::GetBinoRightPMatrix() const {
    return m_binoRightP;
}

void BinoCameraCalibrator::Calibrate(
    const std::vector<std::string> &fileNames,
    std::vector<std::vector<cv::Point3f>> &objectPoints,
    std::vector<std::vector<cv::Point2f>> &imagePoints,
    cv::Matx33f &cameraMatrix, cv::Vec<float, 5> &disCoeffs,
    std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs) {
    cv::Size patternSize(m_checkBoardCol - 1, m_checkBoardRow - 1);

    // Detect feature points
    std::size_t i = 0;
    for (const auto &f : fileNames) {
        std::cout << f << std::endl;

        // 2. Read in the image call cv::findChessboardCornes()
        cv::Mat img = cv::imread(f);
        cv::Mat gray;

        cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

        bool patternFound = cv::findChessboardCorners(
            gray, patternSize, imagePoints[i],
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
                cv::CALIB_CB_FAST_CHECK);

        // 3. Use cv::cornerSubPix() to refine the found corner detections
        if (patternFound) {
            cv::cornerSubPix(gray, imagePoints[i], cv::Size(11, 11),
                             cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS +
                                                  cv::TermCriteria::MAX_ITER,
                                              30, 0.1));
            objectPoints.push_back(m_objp);
        }

        // Display
        cv::drawChessboardCorners(img, patternSize, imagePoints[i],
                                  patternFound);
        cv::imshow("chessboard detection", img);
        cv::waitKey(0);
        ++i;
    }

    int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 +
                cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;
    cv::Size frameSize(1960, 1080);

    std::cout << "Calibrating..." << std::endl;
    // 4. Call "float error = cv::calibrateCamera() with the input coordinates"
    // and output parameters as declared above...

    float error =
        cv::calibrateCamera(objectPoints, imagePoints, frameSize, cameraMatrix,
                            disCoeffs, rvecs, tvecs, flags);

    std::cout << "Reprojection error" << error << "\nK = \n"
              << cameraMatrix << "\nk=\n"
              << disCoeffs << std::endl;
}

void BinoCameraCalibrator::BinoCalibrate() {
    // left camera
    Calibrate(m_leftImageFiles, m_objectPoints, m_leftImagePoints,
              m_leftCameraMatrix, m_leftDistCoeffs, m_leftRVecs, m_leftTVecs);

    m_objectPoints.clear();
    // right camera
    Calibrate(m_rightImageFiles, m_objectPoints, m_rightImagePoints,
              m_rightCameraMatrix, m_rightDistCoeffs, m_rightRVecs,
              m_rightTVecs);

    cv::Size frameSize(1920, 1080);
    float error;
    error = cv::stereoCalibrate(
        m_objectPoints, m_leftImagePoints, m_rightImagePoints,
        m_leftCameraMatrix, m_leftDistCoeffs, m_rightCameraMatrix,
        m_rightDistCoeffs, frameSize, m_binoR, m_binoT, m_binoE, m_binoF,
        cv::CALIB_USE_INTRINSIC_GUESS,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         1e-6));
    std::cout << "stereoCalibrate error: " << error << std::endl;

    // stereo rectify
    std::cout << "stereo rectify..." << std::endl;
    cv::stereoRectify(m_leftCameraMatrix, m_leftDistCoeffs, m_rightCameraMatrix,
                      m_rightDistCoeffs, frameSize, m_binoR, m_binoT,
                      m_binoLeftR, m_binoRightR, m_binoLeftP, m_binoRightP,
                      m_binoQ);

    return;
}

bool BinoCameraCalibrator::SaveAllInfo(const std::string &path) {
    return false;
}
