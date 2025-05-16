#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <algorithm> 
class ImprovedFaceRecognizer {
private:
    std::vector<cv::Mat> trainFaces;
    std::vector<int> trainLabels;
    double confidenceThreshold;

public:
    ImprovedFaceRecognizer() : confidenceThreshold(70.0) {}
    // Methods and techniques
	// Normalized Cross-Correlation (NCC) based face recognizer
    // Pixel-wise Absolute differnce 

    static cv::Ptr<ImprovedFaceRecognizer> create() {
        return cv::makePtr<ImprovedFaceRecognizer>();
    }

    // Train the recognizer with faces and labels
    void train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels) {
        trainFaces = faces;
        trainLabels = labels;
        std::cout << "Training completed with " << faces.size() << " faces." << std::endl;
    }

    // Set confidence threshold
    void setThreshold(double threshold) {
        confidenceThreshold = threshold;
    }

    // Improved prediction using multiple metrics for better accuracy
    void predict(const cv::Mat& face, int& label, double& confidence) {
        if (trainFaces.empty()) {
            label = -1;
            confidence = 999.0;
            return;
        }

        // Use multiple matching methods for better accuracy
        double bestMatch = 0.0;
        int bestLabel = -1;

        for (size_t i = 0; i < trainFaces.size(); i++) {
            // Resize to same dimensions if needed
            cv::Mat resizedTestFace;
            if (face.size() != trainFaces[i].size()) {
                cv::resize(face, resizedTestFace, trainFaces[i].size());
            }
            else {
                resizedTestFace = face;
            }

            // Use multiple comparison methods
            cv::Mat ccorrResult, absDiffResult;

            // Normalized cross-correlation (main method)
            cv::matchTemplate(resizedTestFace, trainFaces[i], ccorrResult, cv::TM_CCORR_NORMED);
            double ccorrValue = ccorrResult.at<float>(0, 0);

            // Absolute difference (secondary method)
            cv::absdiff(resizedTestFace, trainFaces[i], absDiffResult);
            double absDiffValue = 1.0 - (cv::sum(absDiffResult)[0] / (absDiffResult.rows * absDiffResult.cols * 255.0));

            // Combined score (weighted average)
            double combinedScore = (ccorrValue * 0.7) + (absDiffValue * 0.3);

            if (combinedScore > bestMatch) {
                bestMatch = combinedScore;
                bestLabel = trainLabels[i];
            }
        }

        // Convert match score to confidence (lower is better in original API)
        confidence = (1.0 - bestMatch) * 100.0;

        // If confidence is above threshold, mark as unknown
        if (confidence > confidenceThreshold) {
            label = -1;
        }
        else {
            label = bestLabel;
        }
    }
};

class FacialAttendanceSystem {
private:
    cv::CascadeClassifier faceCascade;           // Face detector
    cv::Ptr<ImprovedFaceRecognizer> model;       // Oace recognizer
    std::string dataFolder;                      // Folder to store user face data
    std::string attendanceFile;                  // File to store attendance records
    std::map<int, std::string> userLabels;       // Maps label IDs to user names
    std::vector<int> markedAttendance;           // IDs of users who already marked attendance today
    double recognitionThreshold;                 // Confidence threshold for recognition

    // Helper function to get current date as string (YYYY-MM-DD)
    std::string getCurrentDate() {
        time_t now = time(0);
        struct tm timeInfo;

        // Use safe time functions
#if defined(_WIN32)
        localtime_s(&timeInfo, &now);
#else
        localtime_r(&now, &timeInfo);
#endif

        char dateStr[11]; // YYYY-MM-DD + null terminator
        strftime(dateStr, sizeof(dateStr), "%Y-%m-%d", &timeInfo);
        return std::string(dateStr);
    }

    // Helper function to get current timestamp
    std::string getCurrentTimestamp() {
        time_t now = time(0);
        struct tm timeInfo;

        // Use safe time functions
#if defined(_WIN32)
        localtime_s(&timeInfo, &now);
#else
        localtime_r(&now, &timeInfo);
#endif

        char timeStr[20]; // YYYY-MM-DD HH:MM:SS + null terminator
        strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeInfo);
        return std::string(timeStr);
    }

    // Improved preprocessing for better recognition
    cv::Mat preprocessFace(cv::Mat& frame, const cv::Rect& faceRect) {
        // Extract face with some margin for better recognition
        cv::Rect extendedRect = faceRect;
        // Add 10% margin to each side if possible
        int margin_x = static_cast<int>(faceRect.width * 0.1);
        int margin_y = static_cast<int>(faceRect.height * 0.1);

        extendedRect.x = std::max(0, faceRect.x - margin_x);
        extendedRect.y = std::max(0, faceRect.y - margin_y);
        extendedRect.width = std::min(frame.cols - extendedRect.x, faceRect.width + 2 * margin_x);
        extendedRect.height = std::min(frame.rows - extendedRect.y, faceRect.height + 2 * margin_y);

        cv::Mat face = frame(extendedRect).clone();

        // Convert to grayscale
        cv::Mat grayFace;
        cv::cvtColor(face, grayFace, cv::COLOR_BGR2GRAY);

        // Resize to standard size
        cv::Mat resizedFace;
        cv::resize(grayFace, resizedFace, cv::Size(100, 100));

        // Apply histogram equalization for better contrast
        cv::Mat equalizedFace;
        cv::equalizeHist(resizedFace, equalizedFace);

        // Apply Gaussian blur to reduce noise
        cv::Mat blurredFace;
        cv::GaussianBlur(equalizedFace, blurredFace, cv::Size(3, 3), 0);

        return blurredFace;
    }

    // Load user data from disk
    void loadUserData() {
        userLabels.clear();
        std::string userMapFile = dataFolder + "/users.csv";

        // Create data directory if it doesn't exist (without std::filesystem)
#if defined(_WIN32)
        system(("mkdir " + dataFolder + " 2> nul").c_str());
#else
        system(("mkdir -p " + dataFolder + " 2>/dev/null").c_str());
#endif

        std::ifstream file(userMapFile);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string id_str, name;
                std::getline(ss, id_str, ',');
                std::getline(ss, name);

                try {
                    int id = std::stoi(id_str);
                    userLabels[id] = name;
                    std::cout << "Loaded user: " << id << ", " << name << std::endl;
                }
                catch (const std::exception& e) {
                    std::cerr << "Error parsing user line: " << line << " - " << e.what() << std::endl;
                }
            }
            file.close();
        }

        // Load previously marked attendance for today
        std::string todayDate = getCurrentDate();
        std::ifstream attendanceLog(attendanceFile);
        if (attendanceLog.is_open()) {
            std::string line;
            while (std::getline(attendanceLog, line)) {
                if (line.find(todayDate) != std::string::npos) {
                    std::stringstream ss(line);
                    std::string id_str;
                    std::getline(ss, id_str, ',');
                    try {
                        int id = std::stoi(id_str);
                        markedAttendance.push_back(id);
                        std::cout << "User ID " << id << " already marked attendance today." << std::endl;
                    }
                    catch (...) {
                        // Not an ID, probably a header or incorrectly formatted line
                    }
                }
            }
            attendanceLog.close();
        }
    }

    // Save user data to disk
    void saveUserData() {
        std::string userMapFile = dataFolder + "/users.csv";
        std::ofstream file(userMapFile);
        if (file.is_open()) {
            for (const auto& pair : userLabels) {
                file << pair.first << "," << pair.second << "\n";
            }
            file.close();
            std::cout << "User data saved successfully." << std::endl;
        }
        else {
            std::cerr << "Failed to save user data to " << userMapFile << std::endl;
        }
    }

    // Train face recognition model
    void trainModel() {
        std::vector<cv::Mat> faces;
        std::vector<int> labels;

        // For each user, load face images
        for (const auto& pair : userLabels) {
            int label = pair.first;
            std::string facePath = dataFolder + "/face_" + std::to_string(label) + ".png";

            cv::Mat face = cv::imread(facePath, cv::IMREAD_GRAYSCALE);
            if (!face.empty()) {
                faces.push_back(face);
                labels.push_back(label);
                std::cout << "Loaded face for user " << pair.second << " (ID: " << label << ")" << std::endl;
            }
            else {
                std::cerr << "Warning: Could not load face image for user " << pair.second
                    << " (ID: " << label << ") from " << facePath << std::endl;
            }
        }

        // If we have faces, train the model
        if (!faces.empty()) {
            model->train(faces, labels);
            std::cout << "Face recognition model trained with " << faces.size() << " face(s)." << std::endl;
        }
        else {
            std::cout << "No faces available for training." << std::endl;
        }
    }

    // Mark attendance for a user
    void markAttendance(int userId) {
        // Check if already marked today
        if (std::find(markedAttendance.begin(), markedAttendance.end(), userId) != markedAttendance.end()) {
            std::cout << "Attendance already marked for " << userLabels[userId] << " today." << std::endl;
            return;
        }

        // Mark attendance
        std::string timestamp = getCurrentTimestamp();
        std::ofstream file(attendanceFile, std::ios::app);
        if (file.is_open()) {
            file << userId << "," << userLabels[userId] << "," << timestamp << std::endl;
            file.close();
            std::cout << "Attendance marked for " << userLabels[userId] << " at " << timestamp << std::endl;
            markedAttendance.push_back(userId);
        }
        else {
            std::cerr << "Unable to open attendance file!" << std::endl;
        }
    }

public:
    // Constructor
    FacialAttendanceSystem(const std::string& cascadePath) : recognitionThreshold(50.0) {
        // Load face cascade
        if (!faceCascade.load(cascadePath)) {
            std::cerr << "Error loading face cascade file from: " << cascadePath << std::endl;
            exit(1);
        }
        else {
            std::cout << "Successfully loaded face cascade from: " << cascadePath << std::endl;
        }

        // Create face recognizer with improved threshold
        model = ImprovedFaceRecognizer::create();
        model->setThreshold(recognitionThreshold);

        // Set file paths
        dataFolder = "face_data";
        attendanceFile = "attendance.csv";

        // Create data directory
#if defined(_WIN32)
        system(("mkdir " + dataFolder + " 2> nul").c_str());
#else
        system(("mkdir -p " + dataFolder + " 2>/dev/null").c_str());
#endif

        // Load existing user data
        loadUserData();

        // Train model with existing faces
        trainModel();

        // Create attendance file with header if it doesn't exist
        std::ifstream checkFile(attendanceFile);
        if (!checkFile.good()) {
            std::ofstream createFile(attendanceFile);
            if (createFile.is_open()) {
                createFile << "ID,Name,Timestamp" << std::endl;
                createFile.close();
                std::cout << "Created new attendance file: " << attendanceFile << std::endl;
            }
        }
        checkFile.close();
    }

    // Register a new user with improved face detection
    void registerNewUser() {
        // Open webcam
        cv::VideoCapture capture(0);
        if (!capture.isOpened()) {
            std::cerr << "Error opening webcam!" << std::endl;
            return;
        }

        // Adjust webcam properties for better capture
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture.set(cv::CAP_PROP_BRIGHTNESS, 150);

        // Get user name
        std::string userName;
        std::cout << "Enter your name: ";
        std::getline(std::cin, userName);

        // Generate new user ID
        int newUserId = 1;
        if (!userLabels.empty()) {
            newUserId = userLabels.rbegin()->first + 1;
        }

        std::cout << "Look at the camera. Press SPACE to capture your face or ESC to cancel." << std::endl;

        cv::Mat frame, face;
        bool faceDetected = false;
        int faceQuality = 0;  // Track face detection quality

        while (true) {
            // Capture frame
            capture >> frame;
            if (frame.empty()) break;

            // Make a copy for display
            cv::Mat displayFrame = frame.clone();

            // Convert to grayscale
            cv::Mat grayFrame;
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(grayFrame, grayFrame);

            // Detect faces with improved parameters
            std::vector<cv::Rect> faces;
            faceCascade.detectMultiScale(grayFrame, faces, 1.1, 5, 0, cv::Size(80, 80));

            // Draw rectangle around detected face and assess quality
            faceDetected = !faces.empty();
            faceQuality = 0;

            for (const auto& faceRect : faces) {
                // Check face size (larger is better)
                int size = faceRect.width * faceRect.height;

                // Check face position (centered is better)
                int centerX = faceRect.x + faceRect.width / 2;
                int centerY = faceRect.y + faceRect.height / 2;
                int frameWidth = frame.cols;
                int frameHeight = frame.rows;
                bool isCentered = (abs(centerX - frameWidth / 2) < frameWidth / 4) &&
                    (abs(centerY - frameHeight / 2) < frameHeight / 4);

                // Calculate overall quality score
                if (size > 10000 && isCentered) {
                    faceQuality = 2; // Good
                    cv::rectangle(displayFrame, faceRect, cv::Scalar(0, 255, 0), 2);
                }
                else if (size > 7000) {
                    faceQuality = 1; // Acceptable
                    cv::rectangle(displayFrame, faceRect, cv::Scalar(0, 255, 255), 2);
                }
                else {
                    // Too small
                    cv::rectangle(displayFrame, faceRect, cv::Scalar(0, 0, 255), 2);
                }
            }

            // Display instructions
            std::string message;
            switch (faceQuality) {
            case 2:
                message = "Good face detection! Press SPACE to capture.";
                break;
            case 1:
                message = "Move closer to camera for better quality.";
                break;
            case 0:
                if (faceDetected) {
                    message = "Face too small or off-center. Move closer and center your face.";
                }
                else {
                    message = "No face detected! Please face the camera.";
                }
                break;
            }

            cv::putText(displayFrame, message, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                faceQuality == 2 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

            // Show frame
            cv::imshow("Register New User", displayFrame);

            // Handle key presses
            int key = cv::waitKey(10);
            if (key == 27) { // ESC
                cv::destroyWindow("Register New User");
                capture.release();
                return;
            }
            else if (key == 32 && faceQuality >= 1 && !faces.empty()) { // SPACE
                // Process and save face
                face = preprocessFace(frame, faces[0]);

                // Save the face image
                std::string facePath = dataFolder + "/face_" + std::to_string(newUserId) + ".png";
                bool saveSuccess = cv::imwrite(facePath, face);

                if (saveSuccess) {
                    std::cout << "Face image saved to: " << facePath << std::endl;

                    // Verify the file was saved by trying to read it back
                    cv::Mat testRead = cv::imread(facePath, cv::IMREAD_GRAYSCALE);
                    if (testRead.empty()) {
                        std::cerr << "Warning: Face image save verification failed!" << std::endl;
                    }
                    else {
                        std::cout << "Face image save verified successfully." << std::endl;
                    }

                    // Add user to database
                    userLabels[newUserId] = userName;
                    saveUserData();

                    // Retrain model
                    trainModel();

                    // Mark attendance for new user
                    markAttendance(newUserId);

                    std::cout << "Registration successful! User ID: " << newUserId << std::endl;
                    break;
                }
                else {
                    std::cerr << "Error: Failed to save face image to " << facePath << std::endl;
                    std::cout << "Please try again or check folder permissions." << std::endl;
                }
            }
        }

        capture.release();
        cv::destroyWindow("Register New User");
    }

    // Mark attendance for existing user using improved face recognition
    void markAttendanceForExistingUser() {
        // Check if we have users
        if (userLabels.empty()) {
            std::cout << "No registered users found. Please register a new user first." << std::endl;
            return;
        }

        // Open webcam
        cv::VideoCapture capture(0);
        if (!capture.isOpened()) {
            std::cerr << "Error opening webcam!" << std::endl;
            return;
        }

        // Adjust webcam properties
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        std::cout << "Looking for registered faces. Press ESC to cancel." << std::endl;

        cv::Mat frame;
        bool attendanceMarked = false;
        int consecutiveRecognitions = 0; // Count consecutive successful recognitions of same person
        int lastRecognizedLabel = -1;
        const int requiredConsecutive = 5; // Require 5 consecutive recognitions for confidence

        while (!attendanceMarked) {
            // Capture frame
            capture >> frame;
            if (frame.empty()) break;

            // Make a copy for display
            cv::Mat displayFrame = frame.clone();

            // Convert to grayscale
            cv::Mat grayFrame;
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(grayFrame, grayFrame);

            // Detect faces with improved parameters
            std::vector<cv::Rect> faces;
            faceCascade.detectMultiScale(grayFrame, faces, 1.1, 5, 0, cv::Size(80, 80));

            // Process each detected face
            bool recognized = false;
            int currentLabel = -1;

            for (const auto& faceRect : faces) {
                // Only process faces that are large enough for recognition
                if (faceRect.width < 80 || faceRect.height < 80) {
                    // Face too small, draw orange rectangle
                    cv::rectangle(displayFrame, faceRect, cv::Scalar(0, 165, 255), 2);
                    cv::putText(displayFrame, "Face too small",
                        cv::Point(faceRect.x, faceRect.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 2);
                    continue;
                }

                // Preprocess face for recognition
                cv::Mat face = preprocessFace(frame, faceRect);

                // Try to recognize face
                int predictedLabel = -1;
                double confidence = 0.0;

                try {
                    model->predict(face, predictedLabel, confidence);

                    // Update display based on recognition result
                    std::string displayText;
                    cv::Scalar rectColor;

                    if (predictedLabel != -1) {
                        std::string userName = userLabels[predictedLabel];
                        displayText = userName + " (" + std::to_string(static_cast<int>(confidence)) + "%)";

                        // Track consecutive recognitions
                        if (predictedLabel == lastRecognizedLabel) {
                            consecutiveRecognitions++;
                        }
                        else {
                            consecutiveRecognitions = 1;
                            lastRecognizedLabel = predictedLabel;
                        }

                        // Show progress toward recognition
                        displayText += " " + std::to_string(consecutiveRecognitions) + "/" +
                            std::to_string(requiredConsecutive);

                        // If we've seen the same person enough times, mark attendance
                        if (consecutiveRecognitions >= requiredConsecutive) {
                            recognized = true;
                            currentLabel = predictedLabel;
                            rectColor = cv::Scalar(0, 255, 0); // Green for confirmed
                        }
                        else {
                            rectColor = cv::Scalar(255, 165, 0); // Orange for potential match
                        }
                    }
                    else {
                        displayText = "Unknown (" + std::to_string(static_cast<int>(confidence)) + "%)";
                        consecutiveRecognitions = 0;
                        lastRecognizedLabel = -1;
                        rectColor = cv::Scalar(0, 0, 255); // Red for unknown
                    }

                    // Draw rectangle and text
                    cv::rectangle(displayFrame, faceRect, rectColor, 2);
                    cv::putText(displayFrame, displayText,
                        cv::Point(faceRect.x, faceRect.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, rectColor, 2);

                    // Draw confidence bar
                    int barWidth = 100;
                    int barHeight = 5;
                    int barX = faceRect.x;
                    int barY = faceRect.y + faceRect.height + 10;

                    // Full bar background (gray)
                    cv::rectangle(displayFrame,
                        cv::Rect(barX, barY, barWidth, barHeight),
                        cv::Scalar(150, 150, 150),
                        cv::FILLED);

                    // Confidence level (colored based on value)
                    int confidenceWidth = static_cast<int>((1.0 - confidence / 100.0) * barWidth);
                    cv::Scalar barColor = (confidence < recognitionThreshold) ?
                        cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

                    cv::rectangle(displayFrame,
                        cv::Rect(barX, barY, confidenceWidth, barHeight),
                        barColor,
                        cv::FILLED);
                }
                catch (const cv::Exception& e) {
                    std::cerr << "Error in face recognition: " << e.what() << std::endl;

                    // Draw yellow rectangle for error
                    cv::rectangle(displayFrame, faceRect, cv::Scalar(0, 255, 255), 2);
                    cv::putText(displayFrame, "Recognition Error",
                        cv::Point(faceRect.x, faceRect.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
                }
            }

            // If recognition is confirmed, mark attendance
            if (recognized && currentLabel != -1) {
                markAttendance(currentLabel);
                attendanceMarked = true;

                // Display confirmation message
                std::string confirmMessage = "Attendance marked for " + userLabels[currentLabel];
                cv::putText(displayFrame, confirmMessage,
                    cv::Point(10, frame.rows - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

                // Show the confirmation frame for a moment
                cv::imshow("Mark Attendance", displayFrame);
                cv::waitKey(2000); // Wait 2 seconds to show confirmation
            }

            // Show info
            cv::putText(displayFrame, "Press ESC to cancel", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

            // Show frame count for tracking consecutive recognitions
            if (lastRecognizedLabel != -1) {
                std::string progressText = "Recognition progress: " +
                    std::to_string(consecutiveRecognitions) + "/" +
                    std::to_string(requiredConsecutive);
                cv::putText(displayFrame, progressText, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 165, 0), 2);
            }

            // Show frame
            cv::imshow("Mark Attendance", displayFrame);

            // Handle key presses
            int key = cv::waitKey(10);
            if (key == 27) { // ESC
                break;
            }
        }

        capture.release();
        cv::destroyWindow("Mark Attendance");
    }

    // View attendance records
    void viewAttendanceRecords() {
        std::ifstream file(attendanceFile);
        if (file.is_open()) {
            std::string line;
            std::cout << "\n===== ATTENDANCE RECORDS =====\n";

            // Read and display each line
            while (std::getline(file, line)) {
                std::cout << line << std::endl;
            }
            file.close();
            std::cout << "============================\n";
        }
        else {
            std::cerr << "Unable to open attendance file!" << std::endl;
        }
    }

    // Set confidence threshold
    void setRecognitionThreshold() {
        std::cout << "\nCurrent recognition threshold: " << recognitionThreshold << std::endl;
        std::cout << "Lower values = less strict matching (more false positives)" << std::endl;
        std::cout << "Higher values = more strict matching (more false negatives)" << std::endl;
        std::cout << "Recommended range: 40-70" << std::endl;
        std::cout << "Enter new threshold value (0-100): ";

        std::string input;
        std::getline(std::cin, input);

        try {
            double newThreshold = std::stod(input);
            if (newThreshold >= 0 && newThreshold <= 100) {
                recognitionThreshold = newThreshold;
                model->setThreshold(newThreshold);
                std::cout << "Recognition threshold updated to: " << newThreshold << std::endl;
            }
            else {
                std::cout << "Invalid value. Threshold must be between 0 and 100." << std::endl;
            }
        }
        catch (...) {
            std::cout << "Invalid input. Threshold not changed." << std::endl;
        }
    }

    // Display main menu
    void showMenu() {
        int choice;

        while (true) {
            std::cout << "\n===== FACIAL ATTENDANCE SYSTEM =====\n";
            std::cout << "1. Mark Attendance (Existing User)\n";
            std::cout << "2. Register New User\n";
            std::cout << "3. View Attendance Records\n";
            std::cout << "4. Adjust Recognition Sensitivity\n";
            std::cout << "5. Exit\n";
            std::cout << "Enter your choice: ";

            // Get user choice with validation
            std::string input;
            std::getline(std::cin, input);
            try {
                choice = std::stoi(input);
            }
            catch (...) {
                choice = 0; // Invalid input
            }

            // Process choice
            switch (choice) {
            case 1:
                markAttendanceForExistingUser();
                break;
            case 2 :
                registerNewUser();
                break;
            case 3:
                viewAttendanceRecords();
                break;
            case 4:
                setRecognitionThreshold();
                break;
            case 5:
                std::cout << "Thank you for using Facial Attendance System. Goodbye!\n";
                return;
            default:
                std::cout << "Invalid choice. Please try again.\n";
            }
        }
    }
};

// Helper function to find cascade file
std::string findCascadeFile() {
    std::string cascadePath;
    std::cout << "Enter path to haarcascade_frontalface_default.xml\n";
    std::cout << "(or press Enter to use default location): ";
    std::getline(std::cin, cascadePath);

    // Fix common path issues
    if (!cascadePath.empty()) {
        // Replace single backslashes with forward slashes
        for (size_t i = 0; i < cascadePath.length(); i++) {
            if (cascadePath[i] == '\\') {
                cascadePath[i] = '/';
            }
        }

        // Check if path needs .xml extension
        if (cascadePath.find(".xml") == std::string::npos) {
            cascadePath += ".xml";
        }

        std::cout << "Using path: " << cascadePath << std::endl;
    }

    // Check common locations if user didn't provide a path
    if (cascadePath.empty()) {
        // Try common locations
        std::vector<std::string> commonPaths = {
            "haarcascade_frontalface_default.xml",
            "../haarcascade_frontalface_default.xml",
            "../../haarcascade_frontalface_default.xml",
            "data/haarcascade_frontalface_default.xml",
            // Add OpenCV's common installation paths based on OS
            #ifdef _WIN32
                "C:/opencv/etc/haarcascades/haarcascade_frontalface_default.xml",
                "C:/Program Files/opencv/etc/haarcascades/haarcascade_frontalface_default.xml",
                "C:/Users/New Moon/Desktop/OOP Project/Project/data/haarcascade_frontalface_default.xml"
            #else
                "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
            #endif
        };

        for (const auto& path : commonPaths) {
            std::ifstream testFile(path);
            if (testFile.good()) {
                cascadePath = path;
                std::cout << "Found cascade file at: " << path << std::endl;
                break;
            }
        }

        if (cascadePath.empty()) {
            std::cerr << "Error: Could not find haarcascade_frontalface_default.xml in common locations." << std::endl;
            std::cerr << "Please download it from OpenCV's GitHub repository and provide the correct path." << std::endl;
            std::cerr << "https://github.com/opencv/opencv/tree/master/data/haarcascades" << std::endl;
            exit(1);
        }
    }

    return cascadePath;
}

int main() {
    std::cout << "Welcome to Facial Attendance System!" << std::endl;

    // Find cascade file
    std::string cascadePath = findCascadeFile();

    // Create and run the system
    FacialAttendanceSystem system(cascadePath);
    system.showMenu();

    return 0;
}