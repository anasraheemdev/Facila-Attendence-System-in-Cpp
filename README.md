# Facial Attendance System in CPP

## Overview
Facial Attendance System is a lightweight, console-based application developed in C++ that helps track and manage attendance records. This system allows users to record, modify, and analyze attendance data efficiently and reliably.

![Facial Attendance System](https://www.lips-hci.com/wp-content/uploads/2023/11/facial-recognition.webp)

## Features
- **User Authentication**: Secure login system for administrators
- **Student Management**: Add, update, and remove student records
- **Attendance Tracking**: Mark and update attendance for students
- **Report Generation**: Generate attendance reports with various filters
- **Data Persistence**: Save attendance records in CSV format
- **Simple Interface**: Easy-to-use console-based UI

## Project Structure
```
Facila-Attendence-System-in-Cpp/
├── Project.cpp         # Main source code file
├── Project.sln         # Visual Studio solution file
├── Project.vcxproj     # Visual Studio project file
├── Project.vcxproj.filters # Visual Studio project filters
├── Project.vcxproj.user    # Visual Studio user settings
├── attendance.csv      # CSV file for storing attendance data
├── attendance.txt      # Text backup of attendance data
└── README.md           # This file
```

## Requirements
- C++ compiler (Visual C++ recommended)
- Visual Studio 2019 or later (for .sln and .vcxproj files)
- Standard C++ libraries

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/anasraheemdev/Facila-Attendence-System-in-Cpp.git
   ```
2. Open the solution file (`Project.sln`) in Visual Studio
3. Build the solution (Ctrl+Shift+B)
4. Run the application (F5)

## Usage
1. Launch the application
2. Login with administrator credentials
3. Use the menu options to navigate through different features:
   - Add new students
   - Mark attendance
   - Generate reports
   - Modify existing records
   - Export data to CSV

## Data Format
The system uses CSV format for storing attendance data with the following structure:
- Student ID
- Student Name
- Date
- Status (Present/Absent/Late)
- Additional remarks (if any)

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Created with ❤️ by Anas Raheem

