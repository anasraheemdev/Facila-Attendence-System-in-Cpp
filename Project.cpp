#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <ctime> // Required for time functions

// ===================================================================================
// OOP CONCEPT 1: INHERITANCE AND POLYMORPHISM
// Abstract base class 'FaceRecognizer' defines a contract for any recognizer.
// ===================================================================================
class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;
    virtual void train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels) = 0;
    virtual void predict(const cv::Mat& face, int& label, double& confidence) = 0;
    virtual void setThreshold(double threshold) = 0;
    virtual double getThreshold() const = 0;
};

// ===================================================================================
// Concrete implementation 'CustomFaceRecognizer' inheriting from FaceRecognizer.
// ===================================================================================
class CustomFaceRecognizer : public FaceRecognizer {
private:
    std::vector<cv::Mat> trainFaces;
    std::vector<int> trainLabels;
    double confidenceThreshold;

public:
    CustomFaceRecognizer() : confidenceThreshold(70.0) {}

    void train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels) override {
        trainFaces = faces;
        trainLabels = labels;
        std::cout << "Training completed with " << faces.size() << " faces." << std::endl;
    }

    void setThreshold(double threshold) override {
        confidenceThreshold = threshold;
    }

    double getThreshold() const override {
        return confidenceThreshold;
    }

    void predict(const cv::Mat& face, int& label, double& confidence) override {
        if (trainFaces.empty()) {
            label = -1;
            confidence = 100.0;
            return;
        }

        double bestMatchScore = 0.0;
        int bestLabel = -1;

        for (size_t i = 0; i < trainFaces.size(); ++i) {
            cv::Mat resizedTestFace;
            cv::resize(face, resizedTestFace, trainFaces[i].size());

            cv::Mat ccorrResult;
            cv::matchTemplate(resizedTestFace, trainFaces[i], ccorrResult, cv::TM_CCORR_NORMED);
            double ccorrValue = ccorrResult.at<float>(0, 0);

            cv::Mat absDiffResult;
            cv::absdiff(resizedTestFace, trainFaces[i], absDiffResult);
            double absDiffValue = 1.0 - (cv::sum(absDiffResult)[0] / (absDiffResult.rows * absDiffResult.cols * 255.0));

            double combinedScore = (ccorrValue * 0.7) + (absDiffValue * 0.3);

            if (combinedScore > bestMatchScore) {
                bestMatchScore = combinedScore;
                bestLabel = trainLabels[i];
            }
        }

        confidence = (1.0 - bestMatchScore) * 100.0;

        if (confidence > confidenceThreshold) {
            label = -1;
        }
        else {
            label = bestLabel;
        }
    }
};

// ===================================================================================
// OOP CONCEPT 2: ENCAPSULATION AND ABSTRACTION
// The main class managing the entire system logic.
// ===================================================================================
class FacialAttendanceSystem {
private:
    // --- Private Data Members (Encapsulation) ---
    cv::CascadeClassifier faceCascade;
    std::unique_ptr<FaceRecognizer> model;
    std::string dataFolder;
    std::string attendanceFile;
    std::map<int, std::string> userLabels;
    std::vector<int> markedAttendanceToday;

    // --- Private Helper Methods (Abstraction: Hiding complex logic) ---

    void safe_localtime(const time_t* timer, struct tm* buf) {
#if defined(_WIN32)
        localtime_s(buf, timer);
#else
        localtime_r(timer, buf);
#endif
    }

    std::string getCurrentDate() {
        time_t now = time(0);
        struct tm timeInfo;
        safe_localtime(&now, &timeInfo);
        char dateStr[11];
        strftime(dateStr, sizeof(dateStr), "%Y-%m-%d", &timeInfo);
        return std::string(dateStr);
    }

    std::string getCurrentTimestamp() {
        time_t now = time(0);
        struct tm timeInfo;
        safe_localtime(&now, &timeInfo);
        char timeStr[20];
        strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeInfo);
        return std::string(timeStr);
    }

    void loadUserData() {
        std::string userMapFile = dataFolder + "/users.csv";
        std::ifstream userFile(userMapFile);
        if (userFile.is_open()) {
            std::string line;
            while (std::getline(userFile, line)) {
                std::stringstream ss(line);
                std::string id_str, name;
                if (std::getline(ss, id_str, ',') && std::getline(ss, name)) {
                    try { userLabels[std::stoi(id_str)] = name; }
                    catch (...) {}
                }
            }
            userFile.close();
        }

        std::string todayDate = getCurrentDate();
        std::ifstream attendanceLog(attendanceFile);
        if (attendanceLog.is_open()) {
            std::string line;
            while (std::getline(attendanceLog, line)) {
                if (line.find(todayDate) != std::string::npos) {
                    try {
                        int id = std::stoi(line.substr(0, line.find(',')));
                        markedAttendanceToday.push_back(id);
                    }
                    catch (...) {}
                }
            }
            attendanceLog.close();
        }
    }

    void saveUserData() {
        std::ofstream file(dataFolder + "/users.csv");
        for (const auto& pair : userLabels) {
            file << pair.first << "," << pair.second << "\n";
        }
    }

    void trainModel() {
        std::vector<cv::Mat> faces;
        std::vector<int> labels;
        for (const auto& pair : userLabels) {
            std::string facePath = dataFolder + "/face_" + std::to_string(pair.first) + ".png";
            cv::Mat face = cv::imread(facePath, cv::IMREAD_GRAYSCALE);
            if (!face.empty()) {
                faces.push_back(face);
                labels.push_back(pair.first);
            }
        }
        if (!faces.empty()) model->train(faces, labels);
    }

    cv::Mat preprocessFace(cv::Mat& frame, const cv::Rect& faceRect) {
        cv::Rect extendedRect = faceRect;
        int margin_x = static_cast<int>(faceRect.width * 0.1);
        int margin_y = static_cast<int>(faceRect.height * 0.1);

        extendedRect.x = std::max(0, faceRect.x - margin_x);
        extendedRect.y = std::max(0, faceRect.y - margin_y);
        extendedRect.width = std::min(frame.cols - extendedRect.x, faceRect.width + 2 * margin_x);
        extendedRect.height = std::min(frame.rows - extendedRect.y, faceRect.height + 2 * margin_y);

        cv::Mat face = frame(extendedRect);
        cv::Mat grayFace, resizedFace, equalizedFace, blurredFace;
        cv::cvtColor(face, grayFace, cv::COLOR_BGR2GRAY);
        cv::resize(grayFace, resizedFace, cv::Size(100, 100));
        cv::equalizeHist(resizedFace, equalizedFace);
        cv::GaussianBlur(equalizedFace, blurredFace, cv::Size(3, 3), 0);

        return blurredFace;
    }

    // Returns true if attendance was newly marked, false otherwise.
    bool markAttendance(int userId, const std::string& userName) {
        if (std::find(markedAttendanceToday.begin(), markedAttendanceToday.end(), userId) != markedAttendanceToday.end()) {
            return false; // Already marked
        }

        std::ofstream file(attendanceFile, std::ios::app);
        if (!file.is_open()) {
            return false;
        }

        std::string timestamp = getCurrentTimestamp();
        file << userId << "," << userName << "," << timestamp << std::endl;
        markedAttendanceToday.push_back(userId);
        std::cout << "Attendance marked for " << userName << " at " << timestamp << std::endl;
        return true; // Newly marked
    }

public:
    FacialAttendanceSystem(const std::string& cascadePath) {
        if (!faceCascade.load(cascadePath)) {
            std::cerr << "FATAL ERROR: Could not load face cascade file: " << cascadePath << std::endl;
            exit(1);
        }

        model = std::make_unique<CustomFaceRecognizer>();
        model->setThreshold(50.0);

        dataFolder = "face_data";
        attendanceFile = "attendance.csv";

        system(("mkdir -p " + dataFolder).c_str());
        std::ifstream checkFile(attendanceFile);
        if (!checkFile.good()) {
            std::ofstream createFile(attendanceFile);
            createFile << "ID,Name,Timestamp" << std::endl;
        }
        checkFile.close();

        loadUserData();
        trainModel();
    }

    void run() {
        std::string input;
        int choice = 0;
        do {
            std::cout << "\n===== FACIAL ATTENDANCE SYSTEM =====\n"
                << "1. Mark Attendance (Existing User)\n"
                << "2. Register New User\n"
                << "3. View Attendance Records\n"
                << "4. Adjust Recognition Sensitivity\n"
                << "5. Exit\n"
                << "Enter your choice: ";
            std::getline(std::cin, input);
            try { choice = std::stoi(input); }
            catch (...) { choice = 0; }

            switch (choice) {
            case 1: markAttendanceForExistingUser(); break;
            case 2: registerNewUser(); break;
            case 3: viewAttendanceRecords(); break;
            case 4: setRecognitionThreshold(); break;
            case 5: std::cout << "Goodbye!\n"; break;
            default: std::cout << "Invalid choice. Please try again.\n"; break;
            }
        } while (choice != 5);
    }

    void registerNewUser() {
        std::cout << "Enter your name: ";
        std::string userName;
        std::getline(std::cin, userName);

        cv::VideoCapture capture(0);
        if (!capture.isOpened()) { std::cerr << "Error: Cannot open webcam.\n"; return; }

        std::cout << "Look at the camera. Press SPACE to capture your face, ESC to cancel." << std::endl;

        cv::Mat frame;
        while (true) {
            capture >> frame;
            if (frame.empty()) break;

            cv::Mat displayFrame = frame.clone();
            std::vector<cv::Rect> faces;
            cv::Mat grayFrame;
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(grayFrame, grayFrame);
            faceCascade.detectMultiScale(grayFrame, faces, 1.1, 5, 0, cv::Size(80, 80));

            int faceQuality = 0;
            if (!faces.empty()) {
                int size = faces[0].width * faces[0].height;
                int centerX = faces[0].x + faces[0].width / 2;
                bool isCentered = abs(centerX - frame.cols / 2) < frame.cols / 4;

                if (size > 10000 && isCentered) faceQuality = 2;
                else if (size > 7000) faceQuality = 1;

                cv::Scalar color = (faceQuality == 2) ? cv::Scalar(0, 255, 0) : ((faceQuality == 1) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255));
                cv::rectangle(displayFrame, faces[0], color, 2);
            }

            std::string msg = faceQuality == 2 ? "Good! Press SPACE" : (faceQuality == 1 ? "Move closer" : "No suitable face");
            cv::putText(displayFrame, msg, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Register User", displayFrame);

            int key = cv::waitKey(10);
            if (key == 27) break;
            if (key == 32 && faceQuality > 0) {
                int newUserId = userLabels.empty() ? 1 : userLabels.rbegin()->first + 1;
                cv::Mat processedFace = preprocessFace(frame, faces[0]);

                std::string facePath = dataFolder + "/face_" + std::to_string(newUserId) + ".png";
                if (cv::imwrite(facePath, processedFace)) {
                    userLabels[newUserId] = userName;
                    saveUserData();
                    trainModel();
                    std::cout << "Registration successful for " << userName << " with ID " << newUserId << std::endl;
                }
                else { std::cerr << "Error: Failed to save face image.\n"; }
                break;
            }
        }
        cv::destroyAllWindows();
    }

    // **** THIS IS THE CORRECTED FUNCTION ****
    void markAttendanceForExistingUser() {
        cv::VideoCapture capture(0);
        if (!capture.isOpened()) { std::cerr << "Error: Cannot open webcam.\n"; return; }

        std::cout << "Looking for faces... Press ESC to exit." << std::endl;

        cv::Mat frame;
        while (true) {
            capture >> frame;
            if (frame.empty()) break;

            cv::Mat displayFrame = frame.clone();
            std::vector<cv::Rect> faces;
            cv::Mat grayFrame;
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(grayFrame, grayFrame);
            faceCascade.detectMultiScale(grayFrame, faces, 1.1, 5, 0, cv::Size(80, 80));

            bool shouldBreak = false;
            for (const auto& faceRect : faces) {
                cv::Mat processedFace = preprocessFace(frame, faceRect);
                int label = -1;
                double confidence = 0.0;
                model->predict(processedFace, label, confidence);

                cv::Scalar color = cv::Scalar(0, 0, 255);
                std::string text;

                if (label != -1) {
                    text = userLabels[label];
                    color = cv::Scalar(0, 255, 0);

                    // Check if already marked today
                    if (std::find(markedAttendanceToday.begin(), markedAttendanceToday.end(), label) != markedAttendanceToday.end()) {
                        cv::putText(displayFrame, "ALREADY MARKED", cv::Point(faceRect.x, faceRect.y + faceRect.height + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 100, 0), 2);
                    }
                    else {
                        // If not marked, mark it now
                        if (markAttendance(label, userLabels[label])) {
                            // On successful marking, show a big confirmation message
                            cv::putText(displayFrame, "ATTENDANCE MARKED!", cv::Point(100, displayFrame.rows - 40), cv::FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);
                            shouldBreak = true; // Flag to exit the main loop
                        }
                    }
                }
                else {
                    text = "Unknown";
                }

                cv::rectangle(displayFrame, faceRect, color, 2);
                cv::putText(displayFrame, text, cv::Point(faceRect.x, faceRect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            }

            cv::imshow("Marking Attendance", displayFrame);

            // If attendance was marked, pause for 2 seconds then exit
            if (shouldBreak) {
                cv::waitKey(2000);
                break;
            }

            if (cv::waitKey(10) == 27) break;
        }
        cv::destroyAllWindows();
    }

    void viewAttendanceRecords() {
        std::ifstream file(attendanceFile);
        if (!file.is_open()) { std::cerr << "Could not open attendance file.\n"; return; }
        std::cout << "\n===== ATTENDANCE RECORDS =====\n" << file.rdbuf() << "============================\n";
    }

    void setRecognitionThreshold() {
        std::cout << "\nCurrent recognition threshold: " << model->getThreshold() << std::endl;
        std::cout << "Lower is stricter. Recommended 40-70. Enter new threshold: ";
        double newThreshold;
        std::cin >> newThreshold;
        std::cin.ignore();
        if (newThreshold > 0 && newThreshold <= 100) {
            model->setThreshold(newThreshold);
            std::cout << "Threshold updated to: " << newThreshold << std::endl;
        }
        else {
            std::cout << "Invalid value. Threshold not changed." << std::endl;
        }
    }
};

std::string findCascadeFile() {
    std::vector<std::string> paths = { "haarcascade_frontalface_default.xml", "data/haarcascade_frontalface_default.xml", "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml", "C:/opencv/etc/haarcascades/haarcascade_frontalface_default.xml" };
    for (const auto& path : paths) {
        if (std::ifstream(path).good()) {
            std::cout << "Found cascade file at: " << path << std::endl;
            return path;
        }
    }
    std::cerr << "Error: Could not find haarcascade_frontalface_default.xml.\n";
    return "";
}

int main() {
    std::cout << "Welcome to the Facial Attendance System!" << std::endl;

    std::string cascadePath = findCascadeFile();
    if (cascadePath.empty()) return -1;

    FacialAttendanceSystem system(cascadePath);
    system.run();

    return 0;
}
