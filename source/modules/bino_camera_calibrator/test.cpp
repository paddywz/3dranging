#include <3dranging/bino_camera_calibrator.h>
#include <vector>

int main() {
    std::vector<std::string> leftImageFiles, rightImageFiles;
    int chessBoardCol = 10, chessBoardRow = 7;
    int fieldSize = 21;

    std::vector<cv::Point3f> objp;
    for (int i{1}; i < chessBoardRow; ++i) {
        for (int j{1}; j < chessBoardCol; ++j) {
            objp.push_back(cv::Point3f(j * fieldSize, i * fieldSize, 0));
        }
    }

    BinoCameraCalibrator binoCameraCalibrator(leftImageFiles, rightImageFiles, objp);
}
